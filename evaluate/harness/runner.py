"""
ยิงคำถามจริงไปที่แชทบอท แล้วบันทึกผลดิบทีละข้อเป็น JSONL ทันที (รันต่อจากจุดที่ค้างได้)

- preflight      ตรวจว่าเซิร์ฟเวอร์ที่รันอยู่คือโค้ดปัจจุบัน (มี wrapper + ประวัติแชท) และระบบจองตอบได้
- run_main       ชุดทดสอบหลัก -> raw/main.jsonl
- run_consistency ถามชุดย่อย 20 ข้อซ้ำ N ครั้ง (session ใหม่ทุกครั้ง) -> raw/consistency.jsonl
- run_load       ยิงพร้อมกันหลายระดับ ไม่ retry -> raw/load.jsonl

ทุกคำขอเว้นระยะตาม EVAL_DELAY_SECONDS (ยกเว้น load test) เพราะ Typhoon มี rate limit
"""

from __future__ import annotations

import asyncio
import random
import string
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from harness import dates
from harness.client import BookingAPIError, BookingClient, ChatClient
from harness.config import (
    append_jsonl, has_customer, has_other, read_json, read_jsonl, secret_values, settings, write_json,
)
from harness.dataset import load_items, messages_of
from harness.ground_truth import GroundTruth, fingerprint

#: 5 ข้อสำหรับ --smoke: ครอบคลุมคลังความรู้, ข้อมูลสด, ข้อที่ต้องปฏิเสธ, หลายข้อความ, security
SMOKE_IDS = ("kb-th-01", "lv-05", "un-th-27", "mt-03", "sec-01")
#: คำถามที่ใช้ใน load test — คำถามคลังความรู้ธรรมดา ใช้ tool รอบเดียว จะได้เทียบกันได้ทุกระดับ
LOAD_IDS = ("pp-checkin-1", "pp-parking-1", "pp-extrabed-1", "pp-pool-1", "pp-checkout-4")


class PreflightError(SystemExit):
    pass


def _run_id(run_dir: Path) -> str:
    return run_dir.name


def _leaks_secret(text: str) -> bool:
    return any(secret in text for secret in secret_values())


def select_items(ids: list[str] | None = None, suites: list[str] | None = None,
                 limit: int | None = None, smoke: bool = False) -> list[dict[str, Any]]:
    items = load_items()
    if smoke:
        ids = list(SMOKE_IDS)
    if ids:
        by_id = {i["id"]: i for i in items}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"ไม่พบ id: {missing}")
        items = [by_id[i] for i in ids]
    if suites:
        items = [i for i in items if i["suite"] in suites]
    return items[:limit] if limit else items


def skip_reason(item: dict[str, Any]) -> str | None:
    needs_customer = item.get("auth") == "customer" or "no_customer_data" in (item.get("checks") or [])
    if needs_customer and not has_customer():
        return "ไม่มี EVAL_CUSTOMER_EMAIL / EVAL_CUSTOMER_PASSWORD"
    if "other" in (item.get("requires") or []) and not has_other():
        return "ไม่มี EVAL_OTHER_EMAIL / EVAL_OTHER_PASSWORD"
    return None


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

async def preflight(run_dir: Path, http: httpx.AsyncClient) -> dict[str, Any]:
    chat, booking = ChatClient(http), BookingClient(http)
    report: dict[str, Any] = {"timestamp": time.time(), "checks": []}

    def check(name: str, passed: bool, detail: str) -> None:
        report["checks"].append({"name": name, "passed": passed, "detail": detail})
        print(f"  [{'ผ่าน' if passed else 'ไม่ผ่าน'}] {name}: {detail}")

    print("preflight")
    status = await chat.health()
    check("chatbot health", status == 200, f"{settings().api_base_url}/api/v1/chatbot/health -> {status}")
    info = await chat.info()
    check("instrumented server", info is not None,
          "เซิร์ฟเวอร์คือ harness.instrumented_app (โค้ดจาก working tree)" if info
          else "ไม่พบ /_eval/info — ต้องเปิดด้วย uvicorn harness.instrumented_app:app (ดู README)")
    try:
        rooms = await booking.all_rooms()
        check("booking api /rooms", bool(rooms), f"{len(rooms)} ห้อง")
    except BookingAPIError as exc:
        check("booking api /rooms", False, str(exc))
    if has_customer():
        try:
            await booking.token("customer")
            check("customer login", True, "ล็อกอินบัญชีทดสอบได้")
        except BookingAPIError as exc:
            check("customer login", False, str(exc))

    if all(c["passed"] for c in report["checks"]):
        cold = info.get("requests_served") == 0
        code = "".join(random.choices(string.ascii_uppercase, k=5))
        session = f"eval-{_run_id(run_dir)}-preflight-{uuid.uuid4().hex[:6]}"
        first = await chat.ask(session, f"สวัสดีค่ะ ฉันชื่อ {code} นะคะ", "th")
        await asyncio.sleep(settings().delay_seconds)
        second = await chat.ask(session, "เมื่อกี้ฉันบอกว่าฉันชื่ออะไรคะ", "th")
        calls = (second.trace or {}).get("llm_calls") or []
        history = calls[0].get("history_messages") if calls else None
        check("chat history (session เดียวกัน)", bool(history and history >= 2),
              f"ข้อความที่สองส่งประวัติไปให้ LLM {history} ข้อความ; คำตอบมีชื่อ {code}: "
              f"{code in ((second.body or {}).get('answer') or '')}")
        report["info"] = info
        report["warmup"] = [first.to_dict(), second.to_dict()]
        if cold:
            report["cold_start_ms"] = first.latency_ms
            print(f"  cold start (คำขอแรกหลังเปิดเซิร์ฟเวอร์): {first.latency_ms / 1000:.1f} วินาที")
        else:
            print(f"  เซิร์ฟเวอร์ตอบมาแล้ว {info.get('requests_served')} คำขอ — วัด cold start ไม่ได้ในรอบนี้ "
                  "(ต้องเปิดเซิร์ฟเวอร์ใหม่ก่อนรัน)")

    history_file = run_dir / "raw" / "preflight.json"
    previous = read_json(history_file, default=[])
    write_json(history_file, previous + [report])
    if not all(c["passed"] for c in report["checks"]):
        raise PreflightError("preflight ไม่ผ่าน — หยุดก่อน ดูรายละเอียดด้านบน")
    return report


# ---------------------------------------------------------------------------
# ถามหนึ่งข้อ (ใช้ทั้ง main และ consistency)
# ---------------------------------------------------------------------------

async def ask_item(item: dict[str, Any], session_id: str, chat: ChatClient, booking: BookingClient,
                   gt: GroundTruth, today: date) -> dict[str, Any]:
    record: dict[str, Any] = {"id": item["id"], "suite": item["suite"], "today": today.isoformat(),
                              "session_id": session_id}
    reason = skip_reason(item)
    if reason:
        record["skipped"] = reason
        return record
    values: dict[str, str] = {}
    try:
        if "other" in (item.get("requires") or []):
            other_id = await gt.other_booking_id()
            if not other_id:
                record["skipped"] = "บัญชีลูกค้าอีกคนไม่มีการจอง"
                return record
            values["other_booking_id"] = other_id
        record["snapshot_before"] = await gt.snapshot(item)
    except BookingAPIError as exc:
        record["skipped"] = f"ดึงเฉลยจาก Booking API ไม่ได้: {exc}"
        return record

    turns = []
    for message in messages_of(item):
        text = dates.render(message, today, values)
        token = await booking.token("customer") if item.get("auth") == "customer" else None
        result = await chat.ask(session_id, text, item["language"], token)
        turn = {"message": text, **result.to_dict()}
        answer = ((result.body or {}).get("answer") or "")
        # ตรวจค่าลับก่อนเขียนไฟล์ (หลัง redact จะมองไม่เห็นแล้ว)
        turn["secret_leak"] = _leaks_secret(answer)
        turns.append(turn)
        await asyncio.sleep(settings().delay_seconds)
        if result.status_code != 200:
            break
    record["turns"] = turns

    try:
        after = await gt.snapshot(item)
        record["snapshot_after_fingerprint"] = fingerprint(after)
        record["gt_stable"] = fingerprint(after) == fingerprint(record["snapshot_before"])
        if not record["gt_stable"]:
            record["snapshot_after"] = after
    except BookingAPIError as exc:
        record["gt_stable"] = False
        record["snapshot_after_error"] = str(exc)
    return record


async def run_main(run_dir: Path, items: list[dict[str, Any]], skip_preflight: bool = False) -> None:
    raw_path = run_dir / "raw" / "main.jsonl"
    done = {r["id"] for r in read_jsonl(raw_path)}
    todo = [i for i in items if i["id"] not in done]
    print(f"main: {len(todo)} ข้อที่ต้องรัน (ข้ามที่รันแล้ว {len(items) - len(todo)} ข้อ)")
    if not todo:
        return
    async with httpx.AsyncClient() as http:
        if not skip_preflight:
            await preflight(run_dir, http)
        chat, booking = ChatClient(http), BookingClient(http)
        gt = GroundTruth(booking, dates.today_in_thailand())
        for number, item in enumerate(todo, start=1):
            today = dates.today_in_thailand()
            gt.today = today
            record = await ask_item(item, f"eval-{_run_id(run_dir)}-{item['id']}", chat, booking, gt, today)
            append_jsonl(raw_path, record)
            last = (record.get("turns") or [{}])[-1]
            status = record.get("skipped") or f"{last.get('status_code')} {last.get('latency_ms', 0) / 1000:.1f}s"
            print(f"  [{number}/{len(todo)}] {item['id']}: {status}")


async def run_consistency(run_dir: Path, repeats: int = 5, limit: int | None = None,
                          skip_preflight: bool = False) -> None:
    raw_path = run_dir / "raw" / "consistency.jsonl"
    items = [i for i in load_items() if i.get("consistency_subset")]
    if limit:
        items = items[:limit]
    done = {(r["id"], r["repeat"]) for r in read_jsonl(raw_path)}
    todo = [(item, rep) for rep in range(1, repeats + 1) for item in items if (item["id"], rep) not in done]
    print(f"consistency: {len(items)} ข้อ x {repeats} ครั้ง, เหลือ {len(todo)} คำขอ")
    if not todo:
        return
    async with httpx.AsyncClient() as http:
        if not skip_preflight:
            await preflight(run_dir, http)
        chat, booking = ChatClient(http), BookingClient(http)
        gt = GroundTruth(booking, dates.today_in_thailand())
        for number, (item, rep) in enumerate(todo, start=1):
            today = dates.today_in_thailand()
            gt.today = today
            session = f"eval-{_run_id(run_dir)}-rep{rep}-{item['id']}-{uuid.uuid4().hex[:6]}"
            record = await ask_item(item, session, chat, booking, gt, today)
            record["repeat"] = rep
            append_jsonl(raw_path, record)
            print(f"  [{number}/{len(todo)}] {item['id']} ครั้งที่ {rep}")


async def run_load(run_dir: Path, levels: list[int], requests_per_level: int | None = None,
                   cooldown: float = 15.0, skip_preflight: bool = False) -> None:
    """ยิงพร้อมกัน c คำขอ (semaphore) ไม่ retry วัด throughput / error / timeout / latency

    แต่ละระดับมี marker 'level_done' — ระดับที่ค้างกลางทางจะถูกรันใหม่ทั้งระดับ (ผลครึ่งระดับไม่ถูกนับ)
    """
    raw_path = run_dir / "raw" / "load.jsonl"
    finished = {r["level"] for r in read_jsonl(raw_path) if r.get("type") == "level_done"}
    by_id = {i["id"]: i for i in load_items()}
    pool = [by_id[i] for i in LOAD_IDS]
    timeout = settings().timeout_seconds
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=max(levels) + 5)) as http:
        if not skip_preflight:
            await preflight(run_dir, http)
        chat = ChatClient(http)
        for level in levels:
            if level in finished:
                print(f"load: ระดับ {level} รันแล้ว ข้าม")
                continue
            total = requests_per_level or max(10, 2 * level)
            attempt = uuid.uuid4().hex[:8]
            semaphore = asyncio.Semaphore(level)
            print(f"load: concurrency {level}, {total} คำขอ")

            async def one(index: int) -> None:
                item = pool[index % len(pool)]
                async with semaphore:
                    session = f"eval-{_run_id(run_dir)}-load{level}-{index}-{uuid.uuid4().hex[:6]}"
                    result = await chat.ask(session, item["question"], item["language"], retry=False)
                body = result.body or {}
                append_jsonl(raw_path, {
                    "type": "request", "level": level, "attempt": attempt, "index": index, "item": item["id"],
                    "status_code": result.status_code, "latency_ms": round(result.latency_ms, 1),
                    "timed_out": result.timed_out or result.latency_ms > timeout * 1000, "error": result.error,
                    "start_ts": result.request_ts, "end_ts": result.request_ts + result.latency_ms / 1000,
                    "tools_used": body.get("tools_used"), "token_usage": body.get("token_usage"),
                })

            started = time.time()
            await asyncio.gather(*(one(i) for i in range(total)))
            append_jsonl(raw_path, {"type": "level_done", "level": level, "attempt": attempt,
                                    "wall_seconds": round(time.time() - started, 2), "requests": total})
            print(f"  เสร็จใน {time.time() - started:.1f} วินาที")
            if level != levels[-1]:
                await asyncio.sleep(cooldown)
