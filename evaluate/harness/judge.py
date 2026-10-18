"""
LLM-as-a-Judge — ให้คะแนนคำตอบ 1-5 ตาม rubric และวัด Faithfulness ตามนิยามของ RAGAS

**rubric (แนว G-Eval: ให้เกณฑ์ + ขั้นตอนประเมินชัดเจน แล้วให้ตอบเป็น JSON)**
- correctness     ข้อเท็จจริงในคำตอบตรงกับเฉลยและหลักฐานไหม
- completeness    ตอบครบทุกส่วนที่ลูกค้าถามไหม
- no_fabrication  ไม่มีข้อมูลที่หลักฐานไม่รองรับ (5 = ไม่แต่งเลย)

**Faithfulness (Es et al., 2024)** = จำนวน claim ที่หลักฐานรองรับ / จำนวน claim ทั้งหมด
ขั้นที่ 1 ให้ judge แยกคำตอบเป็น claim ย่อย ขั้นที่ 2 ตรวจทีละ claim กับ context
context คือผลของ tool เต็ม ๆ ที่บอทเห็นในรอบนั้น (จาก trace ของ instrumented_app) ไม่ใช่แค่ sources.snippet

judge เลือกได้ระหว่าง Typhoon กับ Gemini (EVAL_JUDGE_PROVIDER) ถ้าใช้ Typhoon ตัดสิน Typhoon
มีความเสี่ยง self-preference bias (Zheng et al., 2023) — ระบุไว้ในรายงาน และยืนยันด้วยการให้คนติดป้าย 40 ข้อ
แล้วคำนวณ Cohen's kappa (คำสั่ง judge-export / kappa)

ใช้ client ของตัวเอง (ไม่ใช่ app.services.llm_client) เพราะต้องการ temperature 0 และ timeout ยาวกว่าของแอป
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import httpx

from harness.config import append_jsonl, env, read_jsonl, settings, write_json
from harness.metrics import cohen_kappa, percent_agreement, weighted_kappa

DIMENSIONS = ("correctness", "completeness", "no_fabrication")
EVIDENCE_LIMIT = 6000
UNANSWERABLE_REFERENCE = (
    "ระบบไม่มีข้อมูลเรื่องนี้ คำตอบที่ถูกคือแจ้งลูกค้าว่าไม่มีข้อมูลในระบบและแนะนำให้ติดต่อรีสอร์ทโดยตรง "
    "โดยไม่ให้รายละเอียดใด ๆ ที่หลักฐานไม่รองรับ"
)

RUBRIC_SYSTEM = """You are a strict evaluator of a hotel customer-service chatbot for a small resort in Thailand.
Judge ONLY the chatbot's final answer. Use the reference answer and the evidence as ground truth.

Score each criterion from 1 (very poor) to 5 (excellent):
- correctness: Are the facts in the answer consistent with the reference and evidence? Any wrong number, time, price, room name or policy lowers the score sharply.
- completeness: Does the answer cover every part of what the customer asked, as described by the reference?
- no_fabrication: Does the answer avoid information that is not supported by the evidence or reference? 5 = nothing invented; 1 = key details invented.

Evaluation steps:
1. Read the customer's question and the reference answer.
2. Compare each statement in the chatbot answer with the reference and the evidence.
3. Decide each score independently. Politeness, greetings and suggestions to contact the resort are neutral.
4. If the reference says the system has no information, a polite "no information, please contact the resort" deserves 5 on all criteria; inventing an answer deserves 1 on correctness and no_fabrication.

Respond with JSON only: {"correctness": <1-5>, "completeness": <1-5>, "no_fabrication": <1-5>, "reason": "<one or two sentences in Thai>"}"""

CLAIMS_SYSTEM = """Break the chatbot answer into short, self-contained factual claims about the resort, rooms, prices, availability, policies, bookings or nearby places.
Ignore greetings, offers to help, questions back to the customer and generic advice to contact the resort.
Respond with JSON only: {"claims": ["...", "..."]} (at most 12 claims; an empty list if there are none)."""

VERIFY_SYSTEM = """You verify claims against a context. For each claim decide whether it can be directly inferred from the context.
Do not use outside knowledge. Respond with JSON only:
{"verdicts": [{"claim": "...", "supported": true|false}]} in the same order as the claims."""


class JudgeClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http
        cfg = settings()
        self.provider = cfg.judge_provider
        if self.provider == "gemini":
            self.model = cfg.judge_model or env("GEMINI_MODEL")
            self.key = env("GEMINI_API_KEY")
        else:
            self.model = cfg.judge_model or env("TYPHOON_MODEL", "typhoon-v2.5-30b-a3b-instruct")
            self.key = env("TYPHOON_API_KEY")
            self.base_url = env("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1").rstrip("/")
        if not self.key:
            raise SystemExit(f"ไม่มี API key สำหรับ judge ({self.provider})")

    async def complete(self, system: str, user: str) -> tuple[str, dict[str, Any] | None]:
        for attempt in range(settings().max_retries + 1):
            try:
                if self.provider == "gemini":
                    response = await self.http.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                        params={"key": self.key}, timeout=90,
                        json={"systemInstruction": {"parts": [{"text": system}]},
                              "contents": [{"role": "user", "parts": [{"text": user}]}],
                              "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}},
                    )
                else:
                    response = await self.http.post(
                        f"{self.base_url}/chat/completions", timeout=90,
                        headers={"Authorization": f"Bearer {self.key}"},
                        json={"model": self.model, "temperature": 0, "max_tokens": 900,
                              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                    )
            except httpx.HTTPError:
                await asyncio.sleep(min(60, 2 * 2 ** attempt))
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(min(60, 2 * 2 ** attempt))
                continue
            response.raise_for_status()
            data = response.json()
            if self.provider == "gemini":
                text = "".join(p.get("text", "") for p in data["candidates"][0]["content"].get("parts", []))
                usage = data.get("usageMetadata")
            else:
                text = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage")
            return text, usage
        raise RuntimeError("judge เรียก LLM ไม่สำเร็จหลังลองซ้ำ")


def parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def evidence_of(record_turn: dict[str, Any]) -> str:
    calls = ((record_turn or {}).get("trace") or {}).get("tool_calls") or []
    parts = [f"[{c['name']}] {c['output']}" for c in calls]
    return "\n".join(parts)[:EVIDENCE_LIMIT]


def reference_of(item: dict[str, Any], scored: dict[str, Any]) -> str:
    if item.get("answerable") is False:
        return UNANSWERABLE_REFERENCE
    parts = []
    if item.get("reference_answer"):
        parts.append(item["reference_answer"])
    expected = [f["expected"] for f in scored.get("facts") or [] if f["kind"] in ("live", "auth")]
    if expected:
        parts.append("เฉลยจากระบบจอง ณ เวลาทดสอบ: " + "; ".join(expected))
    return "\n".join(parts) or "(ไม่มีคำตอบอ้างอิง ให้ใช้หลักฐานเป็นเกณฑ์)"


async def run(run_dir: Path, items: dict[str, dict[str, Any]], scored: list[dict[str, Any]],
              final_turns: dict[str, dict[str, Any]], limit: int | None = None) -> None:
    raw_path = run_dir / "raw" / "judge.jsonl"
    done = {(r["id"], r["kind"]) for r in read_jsonl(raw_path)}
    targets = [s for s in scored if s.get("status") == "ok" and s.get("answerable") is not None
               and s["category"] != "greeting"]
    if limit:
        targets = targets[:limit]
    async with httpx.AsyncClient() as http:
        client = JudgeClient(http)
        print(f"judge: {client.provider} / {client.model}, {len(targets)} ข้อ")
        for number, s in enumerate(targets, start=1):
            item, turn = items[s["id"]], final_turns.get(s["id"]) or {}
            evidence = evidence_of(turn)
            history = "\n".join(f"ลูกค้า: {m}" for m in s["messages"][:-1])
            if (s["id"], "rubric") not in done:
                user = (
                    (f"Earlier messages in this conversation:\n{history}\n\n" if history else "")
                    + f"Customer question:\n{s['question']}\n\nReference answer:\n{reference_of(item, s)}\n\n"
                    + f"Evidence the chatbot retrieved:\n{evidence or '(none)'}\n\nChatbot answer:\n{s['answer']}"
                )
                text, usage = await client.complete(RUBRIC_SYSTEM, user)
                parsed = parse_json(text) or {}
                scores = {d: parsed.get(d) for d in DIMENSIONS if isinstance(parsed.get(d), int) and 1 <= parsed.get(d) <= 5}
                append_jsonl(raw_path, {"id": s["id"], "kind": "rubric", "provider": client.provider, "model": client.model,
                                        "scores": scores, "reason": parsed.get("reason"), "parse_ok": len(scores) == 3,
                                        "raw_output": text, "usage": usage})
                await asyncio.sleep(settings().delay_seconds)
            if s.get("answered") and evidence and (s["id"], "faithfulness") not in done:
                text, usage1 = await client.complete(CLAIMS_SYSTEM, f"Chatbot answer:\n{s['answer']}")
                claims = [c for c in (parse_json(text) or {}).get("claims", []) if isinstance(c, str)][:12]
                verdicts: list[dict[str, Any]] = []
                usage2 = None
                if claims:
                    await asyncio.sleep(settings().delay_seconds)
                    listing = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
                    text2, usage2 = await client.complete(VERIFY_SYSTEM, f"Context:\n{evidence}\n\nClaims:\n{listing}")
                    verdicts = [v for v in (parse_json(text2) or {}).get("verdicts", []) if isinstance(v, dict)]
                supported = sum(1 for v in verdicts if v.get("supported") is True)
                append_jsonl(raw_path, {
                    "id": s["id"], "kind": "faithfulness", "provider": client.provider, "model": client.model,
                    "claims": claims, "verdicts": verdicts, "supported": supported,
                    "faithfulness": supported / len(claims) if claims and len(verdicts) == len(claims) else None,
                    "usage": [usage1, usage2],
                })
                await asyncio.sleep(settings().delay_seconds)
            print(f"  [{number}/{len(targets)}] {s['id']}")


# ---------------------------------------------------------------------------
# ยืนยันความน่าเชื่อถือของ judge ด้วยป้ายของคน
# ---------------------------------------------------------------------------

HUMAN_COLUMNS = ("id", "suite", "question", "reference", "evidence_excerpt", "answer",
                 "human_correctness", "human_completeness", "human_no_fabrication", "note")


def export_human_labels(run_dir: Path, items: dict[str, dict[str, Any]], scored: list[dict[str, Any]],
                        final_turns: dict[str, dict[str, Any]], n: int = 40, seed: int = 42) -> Path:
    judged = {r["id"] for r in read_jsonl(run_dir / "raw" / "judge.jsonl") if r["kind"] == "rubric" and r.get("parse_ok")}
    pool = sorted((s for s in scored if s["id"] in judged), key=lambda s: s["id"])
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    path = run_dir / "human_labels.csv"
    if path.exists():
        raise SystemExit(f"{path} มีอยู่แล้ว (อาจกรอกไปแล้ว) — ลบเองก่อนถ้าต้องการสุ่มใหม่")
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HUMAN_COLUMNS)
        writer.writeheader()
        for s in sorted(sample, key=lambda s: s["id"]):
            writer.writerow({
                "id": s["id"], "suite": s["suite"], "question": " / ".join(s["messages"]),
                "reference": reference_of(items[s["id"]], s),
                "evidence_excerpt": evidence_of(final_turns.get(s["id"]) or {})[:800],
                "answer": s["answer"],
            })
    return path


def compute_kappa(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "human_labels.csv"
    if not path.exists():
        raise SystemExit(f"ไม่พบ {path} — รัน judge-export ก่อน แล้วกรอกคะแนนช่อง human_*")
    judge = {r["id"]: r["scores"] for r in read_jsonl(run_dir / "raw" / "judge.jsonl") if r["kind"] == "rubric"}
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    result: dict[str, Any] = {"n_rows": len(rows), "dimensions": {}}
    for dim in DIMENSIONS:
        pairs = []
        for row in rows:
            value = (row.get(f"human_{dim}") or "").strip()
            judged = (judge.get(row["id"]) or {}).get(dim)
            if value.isdigit() and 1 <= int(value) <= 5 and judged:
                pairs.append((int(value), int(judged)))
        human, model = [p[0] for p in pairs], [p[1] for p in pairs]
        human_pass, model_pass = [h >= 4 for h in human], [m >= 4 for m in model]
        result["dimensions"][dim] = {
            "n": len(pairs),
            "kappa": cohen_kappa(human, model),
            "weighted_kappa": weighted_kappa(human, model, [1, 2, 3, 4, 5]),
            "agreement": percent_agreement(human, model),
            "pass_kappa": cohen_kappa(human_pass, model_pass),
            "pass_agreement": percent_agreement(human_pass, model_pass),
        }
    write_json(run_dir / "raw" / "kappa.json", result)
    return result
