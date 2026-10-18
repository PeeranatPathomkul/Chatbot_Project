"""
เขียน report.md (ภาษาไทย ฝังกราฟ อ่านจบในไฟล์เดียว) และ report.html (รูปฝังเป็น base64 เปิดได้ทันที)
จาก metrics ที่ aggregate.compute คำนวณไว้ — ไม่เรียก LLM
"""

from __future__ import annotations

import base64
import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def pct(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def num(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:,.{digits}f}"


def secs(ms: float | None) -> str:
    return "–" if ms is None else f"{ms / 1000:.2f} วินาที"


def _table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _img(charts: dict[str, Path], key: str, caption: str) -> str:
    if key not in charts:
        return f"*(ยังไม่มีกราฟ {key} — ยังไม่ได้รันส่วนนี้)*"
    return f"![{caption}](charts/{charts[key].name})\n\n*{caption}*"


def build_markdown(result: dict[str, Any], charts: dict[str, Path]) -> str:
    m = result["metrics"]
    run, ret, tools, ans, ref, perf, rob = (m["run"], m.get("retrieval") or {}, m["tools"], m["answer"], m["refusal"],
                                            m["performance"], m["robustness"])
    server = run.get("server") or {}
    judge = ans["judge"]
    sem = ans["semantic"]
    out: list[str] = []
    add = out.append

    add("# รายงานผลการประเมินแชทบอท พูนสุข รีสอร์ท (Engineering Evaluation)")
    add("")
    add(f"สร้างเมื่อ {datetime.now().strftime('%Y-%m-%d %H:%M')} จากผลดิบใน `{run['run_dir']}/raw/` "
        "ตัวเลขทุกตัวคำนวณใหม่ได้จากผลดิบด้วย `python -m harness report` โดยไม่ต้องเรียก LLM ซ้ำ")
    add("")

    # ---------------- ข้อมูลของรอบ ----------------
    add("## ข้อมูลของรอบที่รัน")
    add("")
    git = run["git"]
    add(_table(["รายการ", "ค่า"], [
        ["วันที่ทดสอบ (เวลาไทย)", ", ".join(run["dates"]) or "–"],
        ["commit", f"`{git['commit']}` ({git['branch']}){' + ไฟล์ที่ยังไม่ commit ' + str(git['dirty_files']) + ' ไฟล์' if git['dirty_files'] else ''}"],
        ["โมเดลของแชทบอท", f"{server.get('llm_provider', '–')} / {server.get('llm_model', '–')}"],
        ["embedding / top_k", f"{server.get('embedding_model', ret.get('embedding_model', '–'))} / top_k = {server.get('retrieval_top_k', ret.get('top_k', '–'))}"],
        ["จำนวนรอบ tool สูงสุด", server.get("agent_max_tool_rounds", "–")],
        ["จำนวน chunk ในคลังความรู้", f"{ret.get('chunk_count', '–')} (ไทย {ret.get('chunks_by_language', {}).get('th', '–')}, อังกฤษ {ret.get('chunks_by_language', {}).get('en', '–')})"],
        ["ชุดทดสอบ", f"{run['items_in_dataset']} ข้อ; รันแล้ว {run['items_run']} ข้อ (สำเร็จ {run['items_ok']}, error {run['items_error']}, ข้าม {run['items_skipped']}); ส่ง {run['messages_sent']} ข้อความ"],
        ["LLM judge", f"{judge.get('provider') or run['judge_provider']} / {judge.get('model') or '–'}"],
    ]))
    if run["skip_reasons"]:
        add("")
        add("ข้อที่ถูกข้าม: " + "; ".join(f"{reason} ({count} ข้อ)" for reason, count in run["skip_reasons"].items()))
    add("")

    # ---------------- ตารางสรุป ----------------
    add("## สรุปผลทุกตัวชี้วัด")
    add("")
    golden = ret.get("golden") or {}
    legacy = ret.get("legacy") or {}
    load = perf["load"]
    rows = [
        ["1. Retrieval", "Hit@1 / Hit@3 / Hit@5 (ระดับ chunk)", f"{pct(golden.get('hit1'))} / {pct(golden.get('hit3'))} / {pct(golden.get('hit5'))}", golden.get("n", "–")],
        ["", "MRR / nDCG@5", f"{num(golden.get('rr'), 3)} / {num(golden.get('ndcg5'), 3)}", golden.get("n", "–")],
        ["", "Hit@1 / Hit@3 ระดับไฟล์ (ชุด 44 ข้อเดิม)", f"{legacy.get('file_hit1_count', '–')}/{legacy.get('n', '–')} / {legacy.get('file_hit3_count', '–')}/{legacy.get('n', '–')}", legacy.get("n", "–")],
        ["2. Tool Calling", "Tool Selection Accuracy (exact set)", pct(tools["selection_exact"]), tools["n"]],
        ["", "Tool Selection Accuracy (ครบ + เรียกเกินเฉพาะที่อนุญาต)", pct(tools["selection_lenient"]), tools["n"]],
        ["", "Argument Accuracy (ทุก field ถูก)", pct(tools["args"]["item_accuracy"]), tools["args"]["n_items"]],
        ["", "Unnecessary Call Rate", pct(tools["unnecessary_call_rate"]), tools["n"]],
        ["", "No-tool rate (คำถามที่ต้องใช้ข้อมูล)", pct(tools["no_tool_rate_when_needed"]), tools["n_needs_data"]],
        ["", "จำนวนรอบเรียก LLM เฉลี่ย / รอบเรียก tool เฉลี่ย", f"{num(tools['avg_llm_calls'])} / {num(tools['avg_tool_rounds'])}", tools["n"]],
        ["3. Answer Correctness", "Key-Fact Accuracy (ระดับข้อเท็จจริง)", pct(ans["fact_accuracy"]), ans["n_facts"]],
        ["", "ข้อที่ครบทุก fact และไม่มีข้อมูลต้องห้าม", pct(ans["complete_and_clean"]), ans["n_complete_items"]],
        ["", "Semantic Similarity (cosine, e5)", num(sem["cosine_mean"], 3), sem["n"]],
        ["", "BERTScore F1 (e5 layer 9, ไม่ rescale)", num(sem["bertscore_f1_mean"], 3), sem["n"]],
        ["", "LLM Judge: ถูกต้อง / ครบถ้วน / ไม่แต่ง (1-5)", " / ".join(num(judge["means"].get(d)) for d in ("correctness", "completeness", "no_fabrication")), judge["n"]],
        ["4. Refusal & Hallucination", "Refusal Precision / Recall / F1", f"{pct(ref['precision'])} / {pct(ref['recall'])} / {pct(ref['f1'])}", ref["n"]],
        ["", "Hallucination Rate", pct(ref["hallucination_rate"]), ref["confusion"]["unanswerable_refused"] + ref["confusion"]["unanswerable_answered"]],
        ["", "False Refusal Rate", pct(ref["false_refusal_rate"]), ref["confusion"]["answerable_refused"] + ref["confusion"]["answerable_answered"]],
        ["", "Faithfulness (RAGAS)", num(judge["faithfulness_mean"], 3), judge["faithfulness_n"]],
        ["5. Performance", "Latency p50 / p95 / p99 (ทุกข้อความ)", f"{secs(perf['latency_overall']['p50'])} / {secs(perf['latency_overall']['p95'])} / {secs(perf['latency_overall']['p99'])}", perf["n_messages"]],
        ["", "Cold start (คำขอแรกหลังเปิดเซิร์ฟเวอร์)", secs(perf["cold_start_ms"]), 1 if perf["cold_start_ms"] else "–"],
        ["", "Token เฉลี่ยต่อคำถาม (prompt / completion / รวม)", f"{num(perf['tokens_mean']['prompt'], 0)} / {num(perf['tokens_mean']['completion'], 0)} / {num(perf['tokens_mean']['total'], 0)}", perf["n_messages"]],
        ["", "ต้นทุนเฉลี่ยต่อคำถาม", f"{perf['cost_per_question']:.5f} {perf['currency']}" if perf["cost_per_question"] is not None else "ยังไม่ได้ตั้งราคา (EVAL_PRICE_*)", perf["n_messages"]],
        ["", "Throughput สูงสุด (load test)", f"{num(max((v['throughput_rps'] or 0 for v in load.values()), default=None))} คำขอ/วินาที" if load else "–", sum(v["requests"] for v in load.values()) if load else "–"],
        ["6. Robustness", "Paraphrase Consistency", pct(rob["paraphrase"]["consistent_rate"]), f"{rob['paraphrase']['groups']} กลุ่ม"],
        ["", f"Run-to-run Consistency ({rob['run_to_run']['repeats']} ครั้ง)", pct(rob["run_to_run"]["same_outcome_rate"]), f"{rob['run_to_run']['items']} ข้อ"],
        ["", "Multi-turn Accuracy", pct(rob["multi_turn"]["accuracy"]), f"{rob['multi_turn']['n']} ชุด"],
        ["", "Attack Success Rate (เป้าหมาย 0%)", pct(rob["security"]["attack_success_rate"]), f"{rob['security']['n']} ข้อ"],
    ]
    add(_table(["ด้าน", "ตัวชี้วัด", "ค่า", "n"], rows))
    add("")

    # ---------------- 1 ----------------
    add("## 1. ความแม่นยำในการค้นคืนข้อมูล (Retrieval Accuracy)")
    add("")
    if ret:
        add("ค้น ChromaDB ตรงด้วย `EmbeddingService` + `VectorStore` ของแอป (prefix `query:`, กรองภาษา, "
            f"top_k = {ret['top_k']}) โดยไม่เรียก LLM chunk ที่ \"เกี่ยวข้อง\" คือ chunk จากไฟล์ใน `expected_sources` "
            "ที่มี key fact ของข้อนั้นอยู่ในเนื้อหา (ข้อที่ไม่มี key fact ใช้ระดับไฟล์) IDCG ของ nDCG@5 คิดจากจำนวน chunk "
            "ที่เกี่ยวข้องทั้งหมดในคลังของภาษานั้น")
        add("")
        add(_table(["ชุด", "n", "Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5", "Hit@1 (ไฟล์)", "Hit@3 (ไฟล์)"], [
            [name, d["n"], pct(d["hit1"]), pct(d["hit3"]), pct(d["hit5"]), num(d["rr"], 3), num(d["ndcg5"], 3), pct(d["file_hit1"]), pct(d["file_hit3"])]
            for name, d in [("Golden รวม", golden), ("Golden ไทย", golden["by_language"]["th"]), ("Golden อังกฤษ", golden["by_language"]["en"]),
                            ("ชุดเดิม 44 ข้อ", legacy)] if d.get("n")
        ]))
        add("")
        add(_table(["หมวด", "n", "Hit@1", "Hit@5", "MRR", "nDCG@5"], [
            [c, d["n"], pct(d["hit1"]), pct(d["hit5"]), num(d["rr"], 3), num(d["ndcg5"], 3)] for c, d in golden["by_category"].items()
        ]))
        add("")
        add(_img(charts, "retrieval_hitk", "ภาพที่ 1 Hit@k, MRR และ nDCG@5 แยกภาษา"))
        add("")
        dist = ret["score_distribution"]
        add(_img(charts, "retrieval_scores", "ภาพที่ 2 การกระจายคะแนน similarity ของ chunk อันดับ 1"))
        add("")
        add(f"**การตีความ:** คะแนนอันดับ 1 ของคำถามที่ตอบได้เฉลี่ย {num(dist['answerable']['mean'], 3)} "
            f"ส่วนคำถามที่ตอบไม่ได้เฉลี่ย {num(dist['unanswerable']['mean'], 3)} ต่างกันเพียง "
            f"{num((dist['answerable']['mean'] or 0) - (dist['unanswerable']['mean'] or 0), 3)} "
            f"threshold ที่ดีที่สุดที่หาได้ ({num(dist['best_threshold']['threshold'], 3)}) แยกสองกลุ่มได้ถูกเพียง "
            f"{pct(dist['best_threshold']['accuracy'])} และมีคำถามที่ตอบไม่ได้ {dist['unanswerable_above_answerable_median']} ข้อ "
            "ที่ได้คะแนนสูงกว่าค่ามัธยฐานของกลุ่มที่ตอบได้ ยืนยันว่าการตัดคำตอบด้วย similarity threshold ใช้ไม่ได้ "
            "ระบบจึงต้องให้ LLM ตัดสินจากเนื้อหา (และวัดผลที่ส่วนที่ 4)")
    else:
        add("*(ยังไม่ได้รัน retrieval)*")
    add("")

    # ---------------- 2 ----------------
    add("## 2. ความถูกต้องของการเรียกใช้เครื่องมือ (Tool Calling Accuracy)")
    add("")
    add("วัดจาก `tools_used` ของ response (ข้อความสุดท้ายของแต่ละข้อ) เทียบกับ `expected_tools` "
        "**exact set** = ชุด tool ตรงกับเฉลยพอดี, **lenient** = เรียกครบตามเฉลยและที่เรียกเกินอยู่ใน `allowed_extra_tools` "
        "Argument Accuracy อ่าน arguments จริงจาก trace ของ `harness.instrumented_app` (ใช้การเรียกครั้งสุดท้ายของ tool ที่เกี่ยวกับวันที่)")
    add("")
    add(_table(["tool", "Precision", "Recall", "F1", "TP", "FP", "FN"], [
        [t, pct(v["precision"]), pct(v["recall"]), pct(v["f1"]), v["tp"], v["fp"], v["fn"]] for t, v in tools["per_tool"].items()
    ]))
    add("")
    add(_table(["argument", "n", "ถูกต้อง"], [[k, v["n"], pct(v["accuracy"])] for k, v in tools["args"]["fields"].items()]))
    add("")
    add(_img(charts, "tool_confusion", "ภาพที่ 3 Confusion matrix ราย tool"))
    add("")
    add(_img(charts, "tool_accuracy", "ภาพที่ 4 Tool Selection Accuracy แยกหมวด"))
    add("")
    add(_img(charts, "tool_precision_recall", "ภาพที่ 5 Precision / Recall ราย tool"))
    add("")
    add(f"**การตีความ:** เลือก tool ตรงเป๊ะ {pct(tools['selection_exact'])} และถ้านับการเรียกเกินที่อนุญาตเป็นถูก "
        f"{pct(tools['selection_lenient'])} คำถามที่ต้องใช้ข้อมูลแต่ไม่เรียก tool เลย {pct(tools['no_tool_rate_when_needed'])} "
        f"เฉลี่ยเรียก LLM {num(tools['avg_llm_calls'])} ครั้งต่อคำถาม (รวมการกระทุ้งให้ตอบใหม่) "
        f"และ arguments ถูกทุก field {pct(tools['args']['item_accuracy'])} ของข้อที่มีเฉลย arguments")
    add("")

    # ---------------- 3 ----------------
    add("## 3. ความถูกต้องของคำตอบ (Answer Correctness)")
    add("")
    add("Key-Fact Accuracy เป็นตัวชี้วัดหลัก (deterministic) เทียบแบบยอมรับรูปแบบเทียบเท่า (14.00 / 14:00 / 2 PM, "
        "คอมมาในตัวเลข, เลขไทย) ข้อเท็จจริงของข้อมูลสดคำนวณจาก Booking API ณ เวลาที่ทดสอบ ข้อที่ข้อมูลเปลี่ยนระหว่างถาม "
        f"(snapshot ก่อน/หลังไม่ตรงกัน) ไม่ถูกนับ: {ans['indeterminate_items']} ข้อ")
    add("")
    add(_table(["แยกตาม", "กลุ่ม", "n", "Key-Fact Accuracy"],
               [["หมวด", k, v["n"], pct(v["rate"])] for k, v in ans["fact_accuracy_by_suite"].items()]
               + [["ภาษา", k, v["n"], pct(v["rate"])] for k, v in ans["fact_accuracy_by_language"].items()]
               + [["แหล่งเฉลย", k, v["n"], pct(v["rate"])] for k, v in ans["fact_accuracy_by_source"].items()]))
    add("")
    add(_img(charts, "keyfact_accuracy", "ภาพที่ 6 Key-Fact Accuracy แยกหมวดและภาษา"))
    add("")
    add(f"Semantic Similarity: cosine เฉลี่ย {num(sem['cosine_mean'], 3)}, BERTScore P/R/F1 = "
        f"{num(sem['bertscore_precision_mean'], 3)} / {num(sem['bertscore_recall_mean'], 3)} / {num(sem['bertscore_f1_mean'], 3)} "
        f"(n={sem['n']}; ใช้ prefix `query:` ทั้งสองฝั่งสำหรับ cosine; BERTScore เขียนเองจาก token embedding ชั้นที่ 9 "
        "ของ multilingual-e5-base ไม่ถ่วง idf และไม่ rescale จึงมีค่าสูงกว่าแพ็กเกจ bert-score และใช้เทียบภายในรายงานนี้เท่านั้น)")
    add("")
    add(_img(charts, "semantic_similarity", "ภาพที่ 7 Semantic Similarity แยกหมวด"))
    add("")
    if judge["n"]:
        add(_table(["หมวด", "n", "ถูกต้อง", "ครบถ้วน", "ไม่แต่ง"], [
            [s, v["n"], num(v["correctness"]), num(v["completeness"]), num(v["no_fabrication"])] for s, v in judge["by_suite"].items()
        ]))
        add("")
        add(_img(charts, "judge_distribution", "ภาพที่ 8 การกระจายคะแนนของ LLM-as-a-Judge"))
        add("")
        if judge["provider"] == "typhoon":
            add("**ข้อควรระวัง:** judge เป็น Typhoon ตัวเดียวกับที่ใช้ตอบ จึงเสี่ยง self-preference bias "
                "(Zheng et al., 2023) ความน่าเชื่อถือของ judge ยืนยันด้วยการเทียบกับคนด้านล่าง")
            add("")
        kappa = judge.get("kappa")
        if kappa:
            add(_table(["มิติ", "n", "Cohen's κ (1-5)", "Weighted κ", "% ตรงกัน (1-5)", "κ (ผ่าน >= 4)", "% ตรงกัน (ผ่าน/ไม่ผ่าน)"], [
                [d, v["n"], num(v["kappa"], 3), num(v["weighted_kappa"], 3), pct(v["agreement"]), num(v["pass_kappa"], 3), pct(v["pass_agreement"])]
                for d, v in kappa["dimensions"].items()
            ]))
        else:
            add("*ยังไม่ได้คำนวณ Cohen's kappa — กรอก `human_labels.csv` (สร้างด้วย `python -m harness judge-export`) "
                "แล้วรัน `python -m harness kappa` และ `python -m harness report`*")
    else:
        add("*(ยังไม่ได้รัน LLM judge)*")
    add("")

    # ---------------- 4 ----------------
    add("## 4. ความสามารถในการปฏิเสธและการป้องกันข้อมูลเท็จ (Refusal and Hallucination)")
    add("")
    c = ref["confusion"]
    add("ใช้ field `answered` ของ response (false = ประโยคปฏิเสธมาตรฐาน) คลาสบวกคือ \"ควรปฏิเสธ\"")
    add("")
    add(_table(["เฉลย \\ ระบบ", "ตอบ", "ปฏิเสธ"], [
        ["มีข้อมูล", c["answerable_answered"], c["answerable_refused"]],
        ["ไม่มีข้อมูล", c["unanswerable_answered"], c["unanswerable_refused"]],
    ]))
    add("")
    add(_img(charts, "refusal", "ภาพที่ 9 Confusion matrix และอัตราการปฏิเสธ"))
    add("")
    add(f"**การตีความ:** ปฏิเสธคำถามที่ไม่มีข้อมูลได้ {c['unanswerable_refused']}/{c['unanswerable_refused'] + c['unanswerable_answered']} "
        f"(Hallucination Rate {pct(ref['hallucination_rate'])}) และปฏิเสธผิดทั้งที่มีข้อมูล {c['answerable_refused']} ข้อ "
        f"(False Refusal Rate {pct(ref['false_refusal_rate'])}) คำตอบของคำถามที่ไม่มีข้อมูลที่มีตัวเลข/ลิงก์ที่แต่งขึ้น "
        f"(ตาม `forbidden`) {pct(ref['fabrication_rate_unanswerable'])} Faithfulness เฉลี่ย {num(ref['faithfulness_mean'], 3)} "
        "(สัดส่วน claim ในคำตอบที่ผลของ tool รองรับ)")
    add("")

    # ---------------- 5 ----------------
    add("## 5. ประสิทธิภาพของระบบ (System Performance)")
    add("")
    add("วัดฝั่ง client ยิงตรงที่ FastAPI (ไม่ผ่าน NestJS) ระหว่างคำขอเว้นระยะตาม `EVAL_DELAY_SECONDS` มี warm-up 2 คำขอ (preflight) ก่อนวัด")
    add("")
    add(_table(["จำนวน tool", "n", "mean", "p50", "p95", "p99", "token เฉลี่ย (prompt/completion/รวม)"], [
        [b, perf["latency_by_tools"][b]["n"], secs(perf["latency_by_tools"][b]["mean"]), secs(perf["latency_by_tools"][b]["p50"]),
         secs(perf["latency_by_tools"][b]["p95"]), secs(perf["latency_by_tools"][b]["p99"]),
         f"{num(perf['tokens_by_tools'][b]['prompt'], 0)} / {num(perf['tokens_by_tools'][b]['completion'], 0)} / {num(perf['tokens_by_tools'][b]['total'], 0)}"]
        for b in ("0", "1", "2", "3+") if perf["latency_by_tools"][b]["n"]
    ] + [["รวม", perf["latency_overall"]["n"], secs(perf["latency_overall"]["mean"]), secs(perf["latency_overall"]["p50"]),
          secs(perf["latency_overall"]["p95"]), secs(perf["latency_overall"]["p99"]),
          f"{num(perf['tokens_mean']['prompt'], 0)} / {num(perf['tokens_mean']['completion'], 0)} / {num(perf['tokens_mean']['total'], 0)}"]]))
    add("")
    add(f"Cold start (คำขอแรกหลังเปิดเซิร์ฟเวอร์ รวมเวลาโหลดโมเดล embedding): {secs(perf['cold_start_ms'])}")
    add("")
    add(_img(charts, "latency_by_tools", "ภาพที่ 10 เวลาตอบสนองแยกตามจำนวน tool"))
    add("")
    add(_img(charts, "tokens_by_tools", "ภาพที่ 11 ปริมาณ token ต่อคำถามแยกตามจำนวน tool"))
    add("")
    if load:
        add(_table(["concurrency", "คำขอ", "สำเร็จ", "error rate", "timeout rate (> 90 วินาที)", "throughput (คำขอ/วินาที)", "p50", "p95", "status code"], [
            [l, v["requests"], v["success"], pct(v["error_rate"]), pct(v["timeout_rate"]), num(v["throughput_rps"], 3),
             secs(v["latency"]["p50"]), secs(v["latency"]["p95"]), ", ".join(f"{k}: {n}" for k, n in v["status_codes"].items())]
            for l, v in sorted(load.items(), key=lambda kv: int(kv[0]))
        ]))
        add("")
        add(_img(charts, "load_test", "ภาพที่ 12 Load test: latency p95, อัตราความผิดพลาด และ throughput"))
    else:
        add("*(ยังไม่ได้รัน load test)*")
    add("")

    # ---------------- 6 ----------------
    add("## 6. ความทนทานของระบบ (Robustness)")
    add("")
    add(_table(["ตัวชี้วัด", "ค่า", "n", "นิยาม"], [
        ["Paraphrase Consistency", pct(rob["paraphrase"]["consistent_rate"]), f"{rob['paraphrase']['groups']} กลุ่ม", "ทุกแบบในกลุ่มผ่าน/ตกเหมือนกัน"],
        ["Paraphrase ผ่านทั้งกลุ่ม", pct(rob["paraphrase"]["all_pass_rate"]), f"{rob['paraphrase']['groups']} กลุ่ม", "ทุกแบบในกลุ่มผ่าน"],
        ["Run-to-run Consistency", pct(rob["run_to_run"]["same_outcome_rate"]), f"{rob['run_to_run']['items']} ข้อ x {rob['run_to_run']['repeats']}", "ผ่าน/ตกเหมือนกันทุกครั้ง (session ใหม่ทุกครั้ง)"],
        ["Run-to-run: answered เหมือนกัน", pct(rob["run_to_run"]["same_answered_rate"]), f"{rob['run_to_run']['items']} ข้อ", "สถานะตอบ/ปฏิเสธเหมือนกันทุกครั้ง"],
        ["Run-to-run: ชุด tool เหมือนกัน", pct(rob["run_to_run"]["same_tools_rate"]), f"{rob['run_to_run']['items']} ข้อ", "เรียก tool ชุดเดียวกันทุกครั้ง"],
        ["Multi-turn Accuracy", pct(rob["multi_turn"]["accuracy"]), f"{rob['multi_turn']['n']} ชุด", "ข้อความสุดท้ายผ่านทุกเกณฑ์ (session_id เดียวกันตลอด)"],
        ["Attack Success Rate", pct(rob["security"]["attack_success_rate"]), f"{rob['security']['n']} ข้อ", "การโจมตีที่สำเร็จ (เป้าหมาย 0%)"],
    ]))
    add("")
    if rob["security"]["by_category"]:
        add(_table(["ประเภทการโจมตี", "n", "สำเร็จ"], [[k, v["n"], pct(v["rate"])] for k, v in rob["security"]["by_category"].items()]))
        add("")
    add(_img(charts, "robustness", "ภาพที่ 13 ตัวชี้วัดความทนทาน"))
    add("")

    # ---------------- เทียบ proposal ----------------
    add("## เปรียบเทียบกับตัวเลขเดิมใน proposal")
    add("")
    base = m["proposal_baseline"]
    add(_table(["รายการ", "ตัวเลขเดิม", "รอบนี้", "หมายเหตุ"], [
        ["Top-1 (ชุด 44 ข้อเดิม, ระดับไฟล์)", base["legacy_top1"], f"{legacy.get('file_hit1_count', '–')}/{legacy.get('n', '–')}", f"top_k ของรอบนี้ = {ret.get('top_k', '–')}"],
        ["Top-3 (ชุด 44 ข้อเดิม, ระดับไฟล์)", base["legacy_top3"], f"{legacy.get('file_hit3_count', '–')}/{legacy.get('n', '–')}", "เทียบเท่า Hit@3 ระดับไฟล์"],
        ["การปฏิเสธคำถามที่ไม่มีข้อมูล", base["refusal"], f"{c['unanswerable_refused']}/{c['unanswerable_refused'] + c['unanswerable_answered']}", "ชุดใหม่ใหญ่ขึ้นและรวมหัวข้อ <<รอเติม>> ทั้งหมด"],
        ["จำนวน chunk", base["chunks"], ret.get("chunk_count", "–"), "คลังความรู้ถูกแยกเป็นสองภาษาและแก้เนื้อหาหลังเขียน proposal"],
        ["เวลาตอบสนอง", base["response_time"], f"p50 {secs(perf['latency_overall']['p50'])}, p95 {secs(perf['latency_overall']['p95'])}",
         "เดิมเป็น RAG รอบเดียว ตอนนี้เป็นลูป tool calling ที่เรียก LLM หลายครั้งและเรียกระบบจองสด"],
    ]))
    add("")

    # ---------------- ข้อจำกัด ----------------
    add("## ข้อจำกัดของการทดสอบ")
    add("")
    for line in [
        "ผล LLM ไม่ deterministic (temperature 0.2) ตัวเลขแต่ละรอบจึงแกว่งได้ ดู Run-to-run Consistency ประกอบ",
        "ข้อมูลสด (ห้องว่าง ราคา การจอง) เปลี่ยนตลอด เฉลยดึงจาก Booking API ตอนรัน ข้อที่ข้อมูลเปลี่ยนระหว่างถามไม่ถูกนับ",
        "Key-fact เป็นการเทียบข้อความ: คำตอบที่ถูกแต่ใช้ถ้อยคำนอกเหนือรายการที่เตรียมไว้อาจถูกนับว่าตก (ใช้ judge และ semantic similarity ประกอบ)",
        "Retrieval ค้นด้วยคำถามของลูกค้าอย่างเดียว ส่วนแอปจริงค้นด้วยคำค้นของโมเดลอีกรอบแล้วรวมผล",
        "Faithfulness ใช้ผลของ tool เต็ม ๆ จาก trace ของ instrumented_app แต่ไม่รวมประวัติแชทและ system prompt ซึ่งโมเดลก็เห็น",
        "LLM judge เป็นโมเดลตระกูลเดียวกับที่ตอบ (ถ้าใช้ Typhoon) อาจลำเอียง ยืนยันด้วย Cohen's kappa กับคน 40 ข้อ",
        "BERTScore คำนวณเองจาก multilingual-e5-base (ไม่ใช่โมเดล/การ rescale มาตรฐานของ bert-score) ใช้เทียบภายในเท่านั้น",
        "Load test ใช้คำถามคลังความรู้ 5 ข้อวนซ้ำ และวัดบนเครื่องพัฒนา (Windows, CPU) ไม่ใช่เซิร์ฟเวอร์จริง ข้อจำกัดหลักคือ rate limit ของ Typhoon",
        "ชุดทดสอบเขียนโดยผู้พัฒนาเอง อาจไม่ครอบคลุมรูปแบบคำถามของลูกค้าจริงทั้งหมด",
    ]:
        add(f"- {line}")
    add("")
    add("## ไฟล์ประกอบ")
    add("")
    add("- `metrics.json` ตัวเลขทุกตัว\n- `tables/items.csv` ผลรายข้อ (ผ่าน/ตก + เหตุผล + คะแนนทุกตัว)\n"
        "- `tables/*.csv` ตารางย่อยแต่ละส่วน\n- `failures.md` ข้อที่ตก พร้อมคำถาม คำตอบ และสิ่งที่ผิด\n"
        "- `raw/*.jsonl` ผลดิบ (คำถาม, response เต็ม, trace, snapshot ของเฉลย, latency, timestamp)\n- `charts/*.png` กราฟทั้งหมด")
    add("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# markdown -> HTML (เฉพาะรูปแบบที่ report ใช้) โดยฝังรูปเป็น base64
# ---------------------------------------------------------------------------

def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def markdown_to_html(markdown: str, base_dir: Path, title: str) -> str:
    body: list[str] = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in block if not re.match(r"^\|(-+\|)+$", r.strip())]
            body.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>"
                        + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</tbody></table>")
            continue
        image = re.match(r"!\[(.*?)\]\((.+?)\)", line)
        if image:
            path = base_dir / image.group(2)
            if path.exists():
                data = base64.b64encode(path.read_bytes()).decode()
                body.append(f'<figure><img alt="{html.escape(image.group(1))}" src="data:image/png;base64,{data}"></figure>')
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            body.append(f"<h{level}>{_inline(line[level:].strip())}</h{level}>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            body.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.startswith("> "):
            body.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.strip():
            body.append(f"<p>{_inline(line)}</p>")
        i += 1
    style = """
    body{font-family:"Leelawadee UI","Tahoma","Noto Sans Thai",sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.6;color:#1f2933}
    h1{border-bottom:3px solid #2f6db5;padding-bottom:6px} h2{margin-top:36px;border-bottom:1px solid #d5dbe1;padding-bottom:4px}
    table{border-collapse:collapse;margin:12px 0;font-size:14px;width:100%} th,td{border:1px solid #cfd6dd;padding:6px 8px;vertical-align:top}
    th{background:#eef3f8;text-align:left} tr:nth-child(even) td{background:#fafbfc}
    figure{margin:16px 0;text-align:center} img{max-width:100%;border:1px solid #e3e8ee}
    code{background:#f1f3f5;padding:1px 4px;border-radius:3px;font-size:90%} em{color:#52606d}
    blockquote{border-left:4px solid #cfd6dd;margin:8px 0;padding:4px 12px;color:#52606d}"""
    return (f'<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{style}</style></head><body>{''.join(body)}</body></html>")


def write(result: dict[str, Any], charts: dict[str, Path], run_dir: Path) -> tuple[Path, Path]:
    markdown = build_markdown(result, charts)
    md_path = run_dir / "report.md"
    md_path.write_text(markdown, encoding="utf-8")
    html_path = run_dir / "report.html"
    html_path.write_text(markdown_to_html(markdown, run_dir, "ผลการประเมินแชทบอท พูนสุข รีสอร์ท"), encoding="utf-8")
    return md_path, html_path
