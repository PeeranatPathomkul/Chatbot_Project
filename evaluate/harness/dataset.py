"""
โหลดและตรวจ Golden Test Set ใน evaluate/datasets/*.yaml

**ทำไมต้องมี validator** — ข้อตกลงของงานนี้คือห้ามแต่งข้อเท็จจริงของรีสอร์ทในเฉลย
การให้คนอ่านตรวจ 160 ข้อด้วยตาไม่พอ validator จึงยืนยันด้วยโค้ดว่า
- ทุก source_ref ชี้ไปยังหัวข้อ/บรรทัดที่มีอยู่จริงใน indexing-pipeline/data/
- ทุก key fact ของข้อที่ตอบจากคลังความรู้พบอยู่ในไฟล์ต้นทาง (เทียบด้วย keyfacts.match_fact ตัวเดียวกับที่ใช้ตรวจคำตอบ)
- ข้อที่ตอบไม่ได้ชี้ไปที่บรรทัด <<รอเติม>> / <<TODO>> จริง และคำนอกคลังไม่มีอยู่ในคลังจริง

วิธีใช้ (รันจากโฟลเดอร์ evaluate/)::

    python -m harness.dataset validate
    python -m harness.dataset sample --ids kb-th-01,lv-05 --out datasets/REVIEW_SAMPLE.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from harness import dates
from harness.keyfacts import fact_label, match_fact

EVAL_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = EVAL_DIR.parent
DATASET_DIR = EVAL_DIR / "datasets"
KNOWLEDGE_DIR = REPO_DIR / "indexing-pipeline" / "data"
TOOLS_FILE = REPO_DIR / "resort-chatbot-api" / "app" / "core" / "tools.py"

LIVE_CHECKS = {
    "availability_answer", "available_names_all", "available_count", "cheapest_price", "total_price",
    "rooms_of_type", "type_price", "rooms_with_amenity", "max_capacity", "guests_plan", "grand_total",
    "restaurant_names_min", "restaurant_match", "promptpay_id", "hold_minutes", "extra_bed_total",
}
#: check ที่ต้องรู้ช่วงวันที่ (ต้องมี live.dates)
DATED_CHECKS = LIVE_CHECKS - {
    "restaurant_names_min", "restaurant_match", "promptpay_id", "hold_minutes", "extra_bed_total",
}
AUTH_CHECKS = {
    "my_current_rooms", "my_next_checkin", "my_next_total", "my_stayed_count",
    "my_payment_status", "my_room_amenity",
}
SECURITY_CHECKS = {
    "no_system_prompt_leak", "no_internal_names", "no_customer_data",
    "no_other_customer_data", "no_secret_leak",
}

#: เกณฑ์ขั้นต่ำจาก EVALUATE_PROMPT.md
MIN_UNANSWERABLE = 40
MIN_MULTI_TURN = 10
MIN_PARAPHRASE_GROUPS = 10
PARAPHRASE_GROUP_SIZE = (3, 5)
MIN_SECURITY = 10
CONSISTENCY_SUBSET_SIZE = 20


def load_items(dataset_dir: Path = DATASET_DIR) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(dataset_dir.glob("*.yaml")):
        for item in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            item["suite"] = path.stem
            items.append(item)
    return items


def known_tools() -> set[str]:
    """ชื่อ tool อ่านจาก TOOL_SCHEMAS ในโค้ดจริง ไม่ได้เขียนรายการซ้ำไว้ที่นี่"""
    return set(re.findall(r'"name":\s*"(\w+)"', TOOLS_FILE.read_text(encoding="utf-8")))


def check_name(check: Any) -> str:
    return next(iter(check)) if isinstance(check, dict) else str(check)


def messages_of(item: dict[str, Any]) -> list[str]:
    return list(item["turns"]) if "turns" in item else [item["question"]]


def _knowledge_lines(rel_path: str) -> list[str] | None:
    path = KNOWLEDGE_DIR / rel_path
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").splitlines()


def _anchor_lines(lines: list[str], anchor: str) -> list[str]:
    """บรรทัดที่ (หลังตัด # ของหัวข้อ) ขึ้นต้นด้วย anchor"""
    return [line for line in lines if line.lstrip("#").strip().startswith(anchor)]


def _validate_item(item: dict[str, Any], tools: set[str], errors: list[str]) -> None:
    iid = item.get("id", "<ไม่มี id>")

    def err(message: str) -> None:
        errors.append(f"{iid}: {message}")

    for field in ("id", "language", "category"):
        if not item.get(field):
            err(f"ไม่มีฟิลด์ {field}")
    if item.get("language") not in ("th", "en"):
        err(f"language ต้องเป็น th/en ได้ {item.get('language')!r}")
    if ("question" in item) == ("turns" in item):
        err("ต้องมี question หรือ turns อย่างใดอย่างหนึ่ง")
    if "answerable" not in item or item["answerable"] not in (True, False, None):
        err("answerable ต้องเป็น true / false / null")

    for field in ("expected_tools", "allowed_extra_tools"):
        unknown = set(item.get(field) or []) - tools
        if unknown:
            err(f"{field} มี tool ที่ไม่มีอยู่จริง: {sorted(unknown)}")

    today = date(2026, 1, 1)
    for message in messages_of(item):
        try:
            dates.render(message, today)
        except ValueError as exc:
            err(str(exc))
        names = dates.template_names(message)
        if "other_booking_id" in names and "other" not in (item.get("requires") or []):
            err("ใช้ {other_booking_id} ต้องมี requires: [other]")
        if names - {"other_booking_id"}:
            err(f"placeholder ไม่รู้จัก: {sorted(names)}")
    for key, value in (item.get("expected_args") or {}).items():
        if key in ("check_in", "check_out") and not dates.is_date_expr(value):
            err(f"expected_args.{key} ต้องเป็นนิพจน์วันที่ ได้ {value!r}")

    live = item.get("live") or {}
    for check in live.get("checks", []):
        name = check_name(check)
        if name not in LIVE_CHECKS:
            err(f"live check ไม่รู้จัก: {name}")
        elif name in DATED_CHECKS and not live.get("dates"):
            err(f"live check {name} ต้องมี live.dates")
    for value in (live.get("dates") or {}).values():
        if not dates.is_date_expr(value):
            err(f"live.dates ไม่ใช่นิพจน์วันที่: {value!r}")
    for check in item.get("auth_checks") or []:
        if check_name(check) not in AUTH_CHECKS:
            err(f"auth check ไม่รู้จัก: {check_name(check)}")
    if item.get("auth_checks") and item.get("auth") != "customer":
        err("auth_checks ต้องใช้คู่กับ auth: customer")
    for check in item.get("checks") or []:
        if check_name(check) not in SECURITY_CHECKS:
            err(f"security check ไม่รู้จัก: {check_name(check)}")
    for pattern in item.get("forbidden") or []:
        try:
            re.compile(pattern)
        except re.error as exc:
            err(f"forbidden regex ผิด {pattern!r}: {exc}")

    for source in item.get("expected_sources") or []:
        if not (KNOWLEDGE_DIR / source).is_file():
            err(f"expected_sources ไม่มีไฟล์ {source}")
        elif not source.startswith(f"{item.get('language')}/") and "turns" not in item:
            err(f"expected_sources {source} คนละภาษากับคำถาม")

    # --- source_ref ต้องชี้ไปที่บรรทัดที่มีจริง ---
    md_texts: list[str] = []
    placeholder_found = False
    for ref in item.get("source_ref") or []:
        if ref.startswith("booking_api:"):
            continue
        rel_path, _, anchor = ref.partition("#")
        lines = _knowledge_lines(rel_path)
        if lines is None:
            err(f"source_ref ไม่มีไฟล์ {rel_path}")
            continue
        md_texts.append("\n".join(lines))
        if anchor:
            matched = _anchor_lines(lines, anchor)
            if not matched:
                err(f"source_ref ไม่พบหัวข้อ/บรรทัด {anchor!r} ใน {rel_path}")
            placeholder_found |= any("<<" in line for line in matched)

    if item.get("answerable") is False and item.get("source_ref"):
        if not placeholder_found:
            err("ข้อที่ตอบไม่ได้ต้องชี้ไปที่บรรทัด <<รอเติม>> / <<TODO>>")
    if item.get("answerable") is False and not item.get("source_ref") and not item.get("absent_terms"):
        err("ข้อที่ตอบไม่ได้ต้องมี source_ref ที่เป็น placeholder หรือ absent_terms")

    # --- key fact ต้องมีอยู่ในไฟล์ต้นทางจริง (ห้ามแต่งเฉลย) ---
    if item.get("key_facts"):
        if not md_texts:
            err("มี key_facts แต่ไม่มี source_ref ที่เป็นไฟล์ในคลังความรู้")
        source_text = "\n".join(md_texts)
        for fact in item["key_facts"]:
            if not match_fact(fact, source_text):
                err(f"key fact {fact_label(fact)!r} ไม่พบในไฟล์ต้นทาง")

    for term in item.get("absent_terms") or []:
        for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
            if term.casefold() in path.read_text(encoding="utf-8").casefold():
                err(f"absent_terms {term!r} มีอยู่ใน {path.relative_to(KNOWLEDGE_DIR).as_posix()}")


def validate(items: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    tools = known_tools()
    for iid, count in Counter(item.get("id") for item in items).items():
        if count > 1:
            errors.append(f"{iid}: id ซ้ำ {count} ครั้ง")
    for item in items:
        _validate_item(item, tools, errors)

    unanswerable = sum(1 for item in items if item.get("answerable") is False)
    if unanswerable < MIN_UNANSWERABLE:
        errors.append(f"คำถามที่ตอบไม่ได้มี {unanswerable} ข้อ ต้อง >= {MIN_UNANSWERABLE}")
    multi_turn = sum(1 for item in items if "turns" in item)
    if multi_turn < MIN_MULTI_TURN:
        errors.append(f"บทสนทนาหลายข้อความมี {multi_turn} ชุด ต้อง >= {MIN_MULTI_TURN}")
    security = sum(1 for item in items if item["suite"] == "security")
    if security < MIN_SECURITY:
        errors.append(f"security มี {security} ข้อ ต้อง >= {MIN_SECURITY}")
    groups: dict[str, int] = defaultdict(int)
    for item in items:
        if item.get("paraphrase_group"):
            groups[item["paraphrase_group"]] += 1
    if len(groups) < MIN_PARAPHRASE_GROUPS:
        errors.append(f"paraphrase มี {len(groups)} กลุ่ม ต้อง >= {MIN_PARAPHRASE_GROUPS}")
    low, high = PARAPHRASE_GROUP_SIZE
    for group, size in groups.items():
        if not low <= size <= high:
            errors.append(f"paraphrase กลุ่ม {group} มี {size} ข้อ ต้อง {low}-{high}")
    subset = sum(1 for item in items if item.get("consistency_subset"))
    if subset != CONSISTENCY_SUBSET_SIZE:
        errors.append(f"consistency_subset มี {subset} ข้อ ต้องเท่ากับ {CONSISTENCY_SUBSET_SIZE}")
    return errors


def summary(items: list[dict[str, Any]]) -> list[str]:
    by_suite = Counter(item["suite"] for item in items)
    by_lang = Counter(item["language"] for item in items)
    turns = sum(len(messages_of(item)) for item in items)
    groups = {item["paraphrase_group"] for item in items if item.get("paraphrase_group")}
    lines = [
        f"ทั้งหมด {len(items)} ข้อ ({turns} ข้อความที่ต้องส่งต่อหนึ่งรอบ) | th {by_lang['th']} / en {by_lang['en']}",
        "suite: " + ", ".join(f"{suite} {count}" for suite, count in sorted(by_suite.items())),
        f"ตอบได้ {sum(1 for i in items if i.get('answerable') is True)} | "
        f"ตอบไม่ได้ {sum(1 for i in items if i.get('answerable') is False)} | "
        f"ไม่นับ (นอกเรื่อง/security) {sum(1 for i in items if i.get('answerable') is None)}",
        f"หลายข้อความ {sum(1 for i in items if 'turns' in i)} ชุด | paraphrase {len(groups)} กลุ่ม | "
        f"ต้องล็อกอิน {sum(1 for i in items if i.get('auth') == 'customer')} ข้อ | "
        f"ชุดย่อย consistency {sum(1 for i in items if i.get('consistency_subset'))} ข้อ",
    ]
    return lines


def _describe_check(check: Any) -> str:
    if isinstance(check, dict):
        name, params = next(iter(check.items()))
        return f"{name}({', '.join(f'{k}={v}' for k, v in params.items())})"
    return str(check)


def render_sample(items: list[dict[str, Any]], ids: list[str], today: date) -> str:
    """ข้อทดสอบที่เลือกในรูปตารางอ่านง่าย พร้อมแปลงวันที่ตามวันนี้ (เวลาไทย)"""
    by_id = {item["id"]: item for item in items}
    out = [
        "# ตัวอย่าง Golden Test Set สำหรับตรวจ",
        "",
        f"วันที่ใช้แปลงนิพจน์วันที่: {today.isoformat()} (เวลาไทย) — ตอนรันจริงจะแปลงตามวันที่รัน",
        "",
    ]
    for number, iid in enumerate(ids, start=1):
        item = by_id[iid]
        messages = [dates.render(m, today) for m in messages_of(item)]
        out.append(f"## {number}. `{iid}` — {item['suite']} / {item['category']} ({item['language']})")
        out.append("")
        if len(messages) == 1:
            out.append(f"**คำถาม:** {messages[0]}")
        else:
            out.append("**บทสนทนา (session เดียวกัน):**")
            out.extend(f"{n}. {m}" for n, m in enumerate(messages, start=1))
        out.append("")
        answerable = {True: "ตอบได้", False: "ต้องปฏิเสธ (answered=false)", None: "ไม่นับในตารางปฏิเสธ"}
        rows = [("answerable", answerable[item.get("answerable")])]
        if item.get("auth"):
            rows.append(("auth", item["auth"]))
        rows.append(("expected_tools", ", ".join(item.get("expected_tools") or []) or "(ไม่เรียก tool)"))
        if item.get("allowed_extra_tools"):
            rows.append(("allowed_extra_tools", ", ".join(item["allowed_extra_tools"])))
        if item.get("expected_args"):
            resolved = dates.resolve_args(item["expected_args"], today)
            rows.append(("expected_args", ", ".join(f"{k}={v}" for k, v in resolved.items())))
        if item.get("expected_sources"):
            rows.append(("expected_sources", ", ".join(item["expected_sources"])))
        if item.get("key_facts"):
            rows.append(("key_facts", " ; ".join(fact_label(f) for f in item["key_facts"])))
        live = item.get("live") or {}
        if live:
            span = ""
            if live.get("dates"):
                resolved = dates.resolve_args(live["dates"], today)
                span = f" ช่วง {resolved['check_in']} ถึง {resolved['check_out']}"
            rows.append(("เฉลยสด (Booking API)" + span, ", ".join(_describe_check(c) for c in live["checks"])))
        if item.get("auth_checks"):
            rows.append(("เฉลยจาก /bookings/me", ", ".join(_describe_check(c) for c in item["auth_checks"])))
        if item.get("checks"):
            rows.append(("security checks", ", ".join(item["checks"])))
        if item.get("forbidden"):
            rows.append(("forbidden (regex)", " ; ".join(f"`{p}`" for p in item["forbidden"])))
        if item.get("absent_terms"):
            rows.append(("absent_terms (ยืนยันว่าไม่มีในคลัง)", ", ".join(item["absent_terms"])))
        if item.get("reference_answer"):
            rows.append(("reference_answer", item["reference_answer"]))
        if item.get("source_ref"):
            rows.append(("source_ref", "<br>".join(item["source_ref"])))
        if item.get("paraphrase_group"):
            rows.append(("paraphrase_group", item["paraphrase_group"]))
        out.append("| ฟิลด์ | ค่า |")
        out.append("|---|---|")
        out.extend(f"| {k} | {str(v).replace('|', '&#124;')} |" for k, v in rows)
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="ตรวจและแสดงตัวอย่าง Golden Test Set")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="ตรวจชุดข้อมูลทั้งหมด")
    sample = sub.add_parser("sample", help="เขียนตัวอย่างข้อทดสอบเป็น markdown")
    sample.add_argument("--ids", required=True, help="id คั่นด้วยคอมมา")
    sample.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    items = load_items()
    if args.command == "validate":
        for line in summary(items):
            print(line)
        errors = validate(items)
        if errors:
            print(f"\nไม่ผ่าน {len(errors)} รายการ:")
            for error in errors:
                print(f"  - {error}")
            return 1
        print("\nผ่านทุกข้อ")
        return 0

    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    missing = [i for i in ids if i not in {item["id"] for item in items}]
    if missing:
        print(f"ไม่พบ id: {missing}")
        return 1
    args.out.write_text(render_sample(items, ids, dates.today_in_thailand()), encoding="utf-8")
    print(f"เขียน {len(ids)} ข้อไปที่ {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
