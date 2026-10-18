"""
คำสั่งของ harness (รันจากโฟลเดอร์ evaluate/ ด้วย python ของ resort-chatbot-api/.venv)

    python -m harness estimate                 ประมาณจำนวนคำขอของแต่ละส่วนก่อนรันจริง
    python -m harness preflight                ตรวจเซิร์ฟเวอร์ (ไม่บันทึกผลรอบใหม่)
    python -m harness retrieval                1. Retrieval (ไม่เรียก LLM)
    python -m harness run [--smoke|--limit N]  2-4, 5 (latency/token), 6 (paraphrase, multi-turn, security)
    python -m harness consistency              6. Run-to-run (ชุดย่อย 20 ข้อ x 5 ครั้ง)
    python -m harness load                     5. Load test (1, 5, 10, 20 พร้อมกัน)
    python -m harness semantic                 3. Cosine + BERTScore (บนเครื่อง ไม่เรียก LLM)
    python -m harness judge                    3-4. LLM-as-a-Judge + Faithfulness
    python -m harness judge-export             สุ่ม 40 ข้อเป็น human_labels.csv ให้คนกรอก
    python -m harness kappa                    Cohen's kappa ระหว่างคนกับ judge (หลังกรอกเสร็จ)
    python -m harness report                   คำนวณ metric + กราฟ + รายงานใหม่จาก raw (ไม่เรียก LLM)
    python -m harness all [--smoke] --yes      รันทุกส่วนตามลำดับ

ทุกคำสั่งรับ --run-dir <ชื่อโฟลเดอร์ใน output/> เพื่อรันต่อ/คำนวณใหม่จากรอบเดิม
(ไม่ระบุ: retrieval/run/all สร้างรอบใหม่, คำสั่งอื่นใช้รอบล่าสุด)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from harness.config import read_jsonl, resolve_run_dir, use_utf8_console
from harness.dataset import load_items, messages_of

#: ประมาณการจำนวนครั้งที่แอปเรียก LLM ต่อคำถาม (ลูป tool + การกระทุ้ง) ใช้เฉพาะตอนประมาณค่าใช้จ่าย
AVG_LLM_CALLS_PER_QUESTION = 2.5


def _print_estimate(smoke: bool, repeats: int, levels: list[int]) -> None:
    from harness.runner import SMOKE_IDS, select_items

    items = select_items(smoke=smoke)
    messages = sum(len(messages_of(i)) for i in items) + 2  # + preflight
    subset = 5 if smoke else sum(1 for i in load_items() if i.get("consistency_subset"))
    consistency = 0 if smoke else subset * repeats
    load = 0 if smoke else sum(max(10, 2 * l) for l in levels)
    judged = sum(1 for i in items if i.get("answerable") is not None and i["category"] != "greeting")
    faith = sum(1 for i in items if i.get("answerable") is True and i["category"] != "greeting")
    judge_calls = judged + 2 * faith
    chat = messages + consistency + load
    print("ประมาณจำนวนคำขอ" + (" (smoke: " + ", ".join(SMOKE_IDS) + ")" if smoke else ""))
    print("  retrieval            0 LLM call (ค้น ChromaDB บนเครื่อง)")
    print(f"  run (ชุดหลัก)        {messages} คำขอแชท  ~ {messages * AVG_LLM_CALLS_PER_QUESTION:.0f} LLM call")
    print(f"  consistency          {consistency} คำขอแชท  ~ {consistency * AVG_LLM_CALLS_PER_QUESTION:.0f} LLM call")
    print(f"  load ({levels})  {load} คำขอแชท  ~ {load * AVG_LLM_CALLS_PER_QUESTION:.0f} LLM call (ยิงพร้อมกันสูงสุด {max(levels)})")
    print(f"  judge                ~ {judge_calls} LLM call (rubric {judged} + faithfulness 2 x {faith})")
    print(f"  รวม                  {chat} คำขอแชท, ~ {chat * AVG_LLM_CALLS_PER_QUESTION + judge_calls:.0f} LLM call")


def _context(run_dir: Path):
    """โหลดของที่ semantic / judge ต้องใช้จากผลของ main run"""
    from harness.aggregate import score_records

    items = {i["id"]: i for i in load_items()}
    records = {r["id"]: r for r in read_jsonl(run_dir / "raw" / "main.jsonl")}
    scored = score_records(list(records.values()), items)
    final_turns = {i: (r.get("turns") or [{}])[-1] for i, r in records.items()}
    return items, scored, final_turns


def cmd_report(run_dir: Path) -> None:
    from harness import aggregate, plots, report

    result = aggregate.compute(run_dir)
    charts = plots.make_all(result, run_dir / "charts")
    font = charts.pop("_font")
    md_path, html_path = report.write(result, charts, run_dir)
    print(f"กราฟ {len(charts)} รูป (ฟอนต์ {font}) -> {run_dir / 'charts'}")
    print(f"รายงาน -> {md_path}\n          {html_path}")


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(prog="python -m harness", description="Evaluation harness ของแชทบอทพูนสุข รีสอร์ท")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run-dir", default=None, help="โฟลเดอร์ใน output/ (ไม่ระบุ = ตามค่าเริ่มต้นของคำสั่ง)")
        return p

    p = add("estimate", "ประมาณจำนวนคำขอ")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--levels", default="1,5,10,20")
    add("preflight", "ตรวจเซิร์ฟเวอร์")
    p = add("retrieval", "ประเมิน retrieval")
    p.add_argument("--limit", type=int)
    p = add("run", "รันชุดทดสอบหลัก")
    p.add_argument("--smoke", action="store_true", help="5 ข้อ end-to-end")
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", help="id คั่นด้วยคอมมา")
    p.add_argument("--suite", help="ชื่อ suite คั่นด้วยคอมมา เช่น kb,live")
    p = add("consistency", "run-to-run consistency")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--limit", type=int)
    p = add("load", "load test")
    p.add_argument("--levels", default="1,5,10,20")
    p.add_argument("--requests", type=int, help="จำนวนคำขอต่อระดับ (ค่าเริ่มต้น max(10, 2 x concurrency))")
    add("semantic", "cosine + BERTScore")
    p = add("judge", "LLM judge + faithfulness")
    p.add_argument("--limit", type=int)
    p = add("judge-export", "สร้าง human_labels.csv")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    add("kappa", "Cohen's kappa คนกับ judge")
    add("report", "สร้าง metrics / กราฟ / รายงานใหม่จาก raw")
    p = add("all", "รันทุกส่วน")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--yes", action="store_true", help="ยืนยันว่าอนุมัติจำนวนคำขอตาม estimate แล้ว")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--levels", default="1,5,10,20")
    p.add_argument("--skip", default="", help="ส่วนที่ข้าม คั่นด้วยคอมมา เช่น load,judge")
    args = parser.parse_args(argv)

    from harness import runner

    if args.command == "estimate":
        _print_estimate(args.smoke, args.repeats, [int(x) for x in args.levels.split(",")])
        return 0

    creates = args.command in ("retrieval", "run", "all") or (args.command == "preflight" and not args.run_dir)
    run_dir = resolve_run_dir(args.run_dir, create=creates)
    print(f"รอบ: {run_dir}")

    if args.command == "preflight":
        import httpx

        async def go():
            async with httpx.AsyncClient() as http:
                await runner.preflight(run_dir, http)
        asyncio.run(go())
    elif args.command == "retrieval":
        from harness import retrieval
        retrieval.run(run_dir, args.limit)
    elif args.command == "run":
        items = runner.select_items(
            ids=[x for x in (args.ids or "").split(",") if x] or None,
            suites=[x for x in (args.suite or "").split(",") if x] or None,
            limit=args.limit, smoke=args.smoke,
        )
        asyncio.run(runner.run_main(run_dir, items))
    elif args.command == "consistency":
        asyncio.run(runner.run_consistency(run_dir, args.repeats, args.limit))
    elif args.command == "load":
        asyncio.run(runner.run_load(run_dir, [int(x) for x in args.levels.split(",")], args.requests))
    elif args.command == "semantic":
        from harness import semantic
        items, scored, _ = _context(run_dir)
        semantic.run(run_dir, scored, {i: it["reference_answer"] for i, it in items.items() if it.get("reference_answer")})
    elif args.command == "judge":
        from harness import judge
        items, scored, final_turns = _context(run_dir)
        asyncio.run(judge.run(run_dir, items, scored, final_turns, args.limit))
    elif args.command == "judge-export":
        from harness import judge
        items, scored, final_turns = _context(run_dir)
        path = judge.export_human_labels(run_dir, items, scored, final_turns, args.n, args.seed)
        print(f"สร้าง {path} แล้ว — กรอกคะแนน 1-5 ในช่อง human_correctness / human_completeness / human_no_fabrication "
              "โดยไม่ดูคะแนนของ judge แล้วรัน `python -m harness kappa`")
    elif args.command == "kappa":
        from harness import judge
        result = judge.compute_kappa(run_dir)
        for dim, v in result["dimensions"].items():
            print(f"  {dim}: n={v['n']} kappa={v['kappa']} weighted={v['weighted_kappa']} agreement={v['agreement']}")
    elif args.command == "report":
        cmd_report(run_dir)
    elif args.command == "all":
        levels = [int(x) for x in args.levels.split(",")]
        skip = {x.strip() for x in args.skip.split(",") if x.strip()}
        _print_estimate(args.smoke, args.repeats, levels)
        if not args.smoke and not args.yes:
            print("\nชุดเต็มใช้ quota ของ LLM ตามตัวเลขข้างบน — ตรวจแล้วรันใหม่พร้อม --yes")
            return 1
        import httpx
        from harness import judge, retrieval, semantic

        async def check():
            async with httpx.AsyncClient() as http:
                await runner.preflight(run_dir, http)
        asyncio.run(check())
        if "retrieval" not in skip:
            retrieval.run(run_dir)
        asyncio.run(runner.run_main(run_dir, runner.select_items(smoke=args.smoke), skip_preflight=True))
        if not args.smoke and "consistency" not in skip:
            asyncio.run(runner.run_consistency(run_dir, args.repeats, skip_preflight=True))
        if not args.smoke and "load" not in skip:
            asyncio.run(runner.run_load(run_dir, levels, skip_preflight=True))
        items, scored, final_turns = _context(run_dir)
        if "semantic" not in skip:
            semantic.run(run_dir, scored, {i: it["reference_answer"] for i, it in items.items() if it.get("reference_answer")})
        if "judge" not in skip:
            asyncio.run(judge.run(run_dir, items, scored, final_turns))
        cmd_report(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
