"""
รวมผลดิบใน raw/ เป็น metrics.json + tables/*.csv + failures.md — ไม่เรียก LLM และไม่เรียก network

รันซ้ำได้ทุกเมื่อ (`python -m harness report`) เช่น หลังแก้นิยาม metric หรือหลังกรอก human_labels.csv
"""

from __future__ import annotations

import csv
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from harness.config import REPO_DIR, read_json, read_jsonl, settings, write_json
from harness.dataset import load_items
from harness.judge import DIMENSIONS
from harness.keyfacts import match_fact
from harness.metrics import (
    hit_at_k, latency_summary, mean, ndcg_at_k, percentile, precision_recall_f1, reciprocal_rank,
    refusal_metrics, safe_div,
)
from harness.scoring import score_item

PROPOSAL_BASELINE = {
    "legacy_top1": "44/44", "legacy_top3": "44/44", "refusal": "6/6", "chunks": 113, "response_time": "0.7-1.0 วินาที",
}


def _latest_by(records: list[dict[str, Any]], key) -> dict[Any, dict[str, Any]]:
    latest: dict[Any, dict[str, Any]] = {}
    for record in records:
        latest[key(record)] = record
    return latest


def _rate(passed: int, total: int) -> float | None:
    return safe_div(passed, total)


def _group_rate(rows: list[dict[str, Any]], key: str, field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            groups[str(row.get(key))].append(bool(row[field]))
    return {g: {"n": len(v), "rate": _rate(sum(v), len(v))} for g, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# 1. Retrieval
# ---------------------------------------------------------------------------

def _chunk_relevant(chunk_source: str, chunk_text: str, query: dict[str, Any]) -> bool:
    if chunk_source not in query["expected_sources"]:
        return False
    facts = query.get("key_facts") or []
    return any(match_fact(f, chunk_text) for f in facts) if facts else True


def retrieval_metrics(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = read_json(run_dir / "raw" / "retrieval_meta.json")
    records = list(_latest_by(read_jsonl(run_dir / "raw" / "retrieval.jsonl"), lambda r: r["id"]).values())
    if not meta or not records:
        return {}, []
    chunks = meta["chunks"]
    rows = []
    for q in records:
        hits = q["hits"]
        file_rel = [h["source"] in q["expected_sources"] for h in hits]
        chunk_rel = [_chunk_relevant(h["source"], h["text"], q) for h in hits]
        total_relevant = sum(1 for c in chunks if c["language"] == q["language"] and _chunk_relevant(c["source"], c["text"], q))
        rows.append({
            "set": q["set"], "id": q["id"], "suite": q.get("suite", "legacy"), "category": q.get("category", "legacy"),
            "language": q["language"], "answerable": q["answerable"], "question": q["question"],
            "expected_sources": ";".join(q["expected_sources"]),
            "top_sources": ";".join(h["source"] for h in hits),
            "top1_similarity": hits[0]["similarity"] if hits else None,
            "hit1": hit_at_k(chunk_rel, 1), "hit3": hit_at_k(chunk_rel, 3), "hit5": hit_at_k(chunk_rel, 5),
            "rr": reciprocal_rank(chunk_rel, 5), "ndcg5": ndcg_at_k(chunk_rel, 5, total_relevant),
            "file_hit1": hit_at_k(file_rel, 1), "file_hit3": hit_at_k(file_rel, 3), "file_hit5": hit_at_k(file_rel, 5),
            "total_relevant_chunks": total_relevant, "query_ms": q.get("query_ms"),
        })

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            **{m: mean([r[m] for r in subset]) for m in ("hit1", "hit3", "hit5", "rr", "ndcg5", "file_hit1", "file_hit3", "file_hit5")},
        }

    golden = [r for r in rows if r["set"] == "golden" and r["answerable"] and r["expected_sources"]]
    legacy = [r for r in rows if r["set"] == "legacy" and r["answerable"]]
    answerable_scores = [r["top1_similarity"] for r in rows if r["answerable"] and r["top1_similarity"] is not None]
    unanswerable_scores = [r["top1_similarity"] for r in rows if r["answerable"] is False and r["top1_similarity"] is not None]

    # threshold ที่ดีที่สุดที่แยกสองกลุ่มได้ — ถ้ายังผิดเยอะ แปลว่าใช้ threshold ตัดคำตอบไม่ได้
    best = {"threshold": None, "accuracy": None}
    for t in sorted(set(answerable_scores + unanswerable_scores)):
        correct = sum(1 for s in answerable_scores if s >= t) + sum(1 for s in unanswerable_scores if s < t)
        acc = correct / (len(answerable_scores) + len(unanswerable_scores))
        if best["accuracy"] is None or acc > best["accuracy"]:
            best = {"threshold": t, "accuracy": acc}

    metrics = {
        "chunk_count": meta["chunk_count"],
        "chunks_by_language": meta["chunks_by_language"],
        "top_k": meta["top_k"],
        "embedding_model": meta["embedding_model"],
        "golden": {**summarize(golden), "by_language": {l: summarize([r for r in golden if r["language"] == l]) for l in ("th", "en")},
                   "by_category": {c: summarize([r for r in golden if r["category"] == c]) for c in sorted({r["category"] for r in golden})}},
        "legacy": {**summarize(legacy), "by_language": {l: summarize([r for r in legacy if r["language"] == l]) for l in ("th", "en")},
                   "file_hit1_count": int(sum(r["file_hit1"] for r in legacy)), "file_hit3_count": int(sum(r["file_hit3"] for r in legacy))},
        "score_distribution": {
            "answerable": {"n": len(answerable_scores), "mean": mean(answerable_scores), "median": percentile(answerable_scores, 50),
                           "min": min(answerable_scores, default=None), "max": max(answerable_scores, default=None)},
            "unanswerable": {"n": len(unanswerable_scores), "mean": mean(unanswerable_scores), "median": percentile(unanswerable_scores, 50),
                             "min": min(unanswerable_scores, default=None), "max": max(unanswerable_scores, default=None)},
            "best_threshold": best,
            "unanswerable_above_answerable_median": sum(1 for s in unanswerable_scores if s >= (percentile(answerable_scores, 50) or 1)),
        },
        "answerable_scores": answerable_scores,
        "unanswerable_scores": unanswerable_scores,
    }
    return metrics, rows


# ---------------------------------------------------------------------------
# 2-4. Tool calling, Answer correctness, Refusal (จาก main run)
# ---------------------------------------------------------------------------

def score_records(records: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [score_item(items[r["id"]], r) for r in records if r["id"] in items]


def tool_metrics(scored: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [s for s in scored if s["status"] == "ok" and "expected_tools" in items[s["id"]]]
    tools = sorted({t for s in rows for t in s["expected_tools"] + s["tools_used"] + s["allowed_extra_tools"]})
    per_tool = {}
    for tool in tools:
        tp = sum(1 for s in rows if tool in s["expected_tools"] and tool in s["tools_used"])
        fn = sum(1 for s in rows if tool in s["expected_tools"] and tool not in s["tools_used"])
        fp = sum(1 for s in rows if tool in s["tools_used"] and tool not in s["expected_tools"] and tool not in s["allowed_extra_tools"])
        per_tool[tool] = precision_recall_f1(tp, fp, fn)
    labels = tools + ["(ไม่เรียก)"]
    matrix = {e: {c: 0 for c in labels} for e in labels}
    for s in rows:
        called = set(s["tools_used"]) or {"(ไม่เรียก)"}
        for e in s["expected_tools"] or ["(ไม่เรียก)"]:
            for c in called:
                matrix[e][c] += 1
    needs_data = [s for s in rows if s["expected_tools"]]
    arg_rows = [s for s in rows if s.get("args")]
    fields: dict[str, list[bool]] = defaultdict(list)
    for s in arg_rows:
        for name, f in s["args"]["fields"].items():
            fields[name].append(f["correct"])
    return {
        "n": len(rows),
        "selection_exact": _rate(sum(s["tool_exact"] for s in rows), len(rows)),
        "selection_lenient": _rate(sum(s["tool_lenient"] for s in rows), len(rows)),
        "selection_exact_by_suite": _group_rate(rows, "suite", "tool_exact"),
        "selection_lenient_by_suite": _group_rate(rows, "suite", "tool_lenient"),
        "selection_lenient_by_language": _group_rate(rows, "language", "tool_lenient"),
        "per_tool": per_tool,
        "confusion_labels": labels,
        "confusion": matrix,
        "unnecessary_call_rate": _rate(sum(1 for s in rows if s["unnecessary_tools"]), len(rows)),
        "no_tool_rate_when_needed": _rate(sum(1 for s in needs_data if not s["tools_used"]), len(needs_data)),
        "n_needs_data": len(needs_data),
        "avg_llm_calls": mean([s["llm_calls"] for s in rows if s["llm_calls"]]),
        "avg_tool_rounds": mean([s["tool_rounds"] for s in rows if s["llm_calls"]]),
        "avg_tools_per_question": mean([s["tool_count"] for s in rows]),
        "args": {
            "n_items": len(arg_rows),
            "item_accuracy": _rate(sum(s["args"]["all_correct"] for s in arg_rows), len(arg_rows)),
            "fields": {name: {"n": len(v), "accuracy": _rate(sum(v), len(v))} for name, v in sorted(fields.items())},
        },
    }


def answer_metrics(scored: list[dict[str, Any]], semantic: dict[str, dict[str, Any]],
                   judge: dict[tuple[str, str], dict[str, Any]], kappa: dict[str, Any] | None) -> dict[str, Any]:
    ok = [s for s in scored if s["status"] == "ok"]
    facts = [(s, f) for s in ok for f in s["facts"] if f["passed"] is not None]
    by: dict[str, dict[str, list[bool]]] = {"suite": defaultdict(list), "language": defaultdict(list), "kind": defaultdict(list)}
    for s, f in facts:
        by["suite"][s["suite"]].append(f["passed"])
        by["language"][s["language"]].append(f["passed"])
        by["kind"][f["kind"]].append(f["passed"])
    with_facts = [s for s in ok if s["answerable"] is True and s["facts"]]
    determinable = [s for s in with_facts if s["item_pass"] is not None]
    sem_rows = [semantic[s["id"]] for s in ok if s["id"] in semantic]
    rubric = [judge[(s["id"], "rubric")] for s in ok if (s["id"], "rubric") in judge and judge[(s["id"], "rubric")].get("parse_ok")]
    faith = [judge[(s["id"], "faithfulness")]["faithfulness"] for s in ok
             if (s["id"], "faithfulness") in judge and judge[(s["id"], "faithfulness")].get("faithfulness") is not None]
    rubric_by_suite: dict[str, dict[str, Any]] = {}
    for suite in sorted({s["suite"] for s in ok}):
        scores = [judge[(s["id"], "rubric")]["scores"] for s in ok if s["suite"] == suite
                  and (s["id"], "rubric") in judge and judge[(s["id"], "rubric")].get("parse_ok")]
        if scores:
            rubric_by_suite[suite] = {d: mean([x[d] for x in scores]) for d in DIMENSIONS} | {"n": len(scores)}
    sem_by_suite = {}
    for suite in sorted({s["suite"] for s in ok}):
        values = [semantic[s["id"]] for s in ok if s["suite"] == suite and s["id"] in semantic]
        if values:
            sem_by_suite[suite] = {"n": len(values), "cosine": mean([v["cosine"] for v in values]),
                                   "bertscore_f1": mean([v["bertscore_f1"] for v in values])}
    return {
        "fact_accuracy": _rate(sum(f["passed"] for _, f in facts), len(facts)),
        "n_facts": len(facts),
        "fact_accuracy_by_suite": {k: {"n": len(v), "rate": _rate(sum(v), len(v))} for k, v in sorted(by["suite"].items())},
        "fact_accuracy_by_language": {k: {"n": len(v), "rate": _rate(sum(v), len(v))} for k, v in sorted(by["language"].items())},
        "fact_accuracy_by_source": {k: {"n": len(v), "rate": _rate(sum(v), len(v))} for k, v in sorted(by["kind"].items())},
        "complete_and_clean": _rate(sum(1 for s in determinable if s["item_pass"]), len(determinable)),
        "n_complete_items": len(determinable),
        "indeterminate_items": len(with_facts) - len(determinable),
        "forbidden_hit_items": sum(1 for s in ok if s["forbidden_hits"]),
        "fake_room_items": sum(1 for s in ok if s["fake_rooms"]),
        "semantic": {
            "n": len(sem_rows),
            "cosine_mean": mean([r["cosine"] for r in sem_rows]),
            "bertscore_precision_mean": mean([r["bertscore_precision"] for r in sem_rows]),
            "bertscore_recall_mean": mean([r["bertscore_recall"] for r in sem_rows]),
            "bertscore_f1_mean": mean([r["bertscore_f1"] for r in sem_rows]),
            "by_suite": sem_by_suite,
        },
        "judge": {
            "n": len(rubric),
            "provider": rubric[0]["provider"] if rubric else None,
            "model": rubric[0]["model"] if rubric else None,
            "means": {d: mean([r["scores"][d] for r in rubric]) for d in DIMENSIONS},
            "distribution": {d: {str(k): sum(1 for r in rubric if r["scores"][d] == k) for k in range(1, 6)} for d in DIMENSIONS},
            "by_suite": rubric_by_suite,
            "parse_failures": sum(1 for (i, k), r in judge.items() if k == "rubric" and not r.get("parse_ok")),
            "faithfulness_mean": mean(faith),
            "faithfulness_n": len(faith),
            "kappa": kappa,
        },
    }


def refusal_section(scored: list[dict[str, Any]], judge: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    ok = [s for s in scored if s["status"] == "ok" and s["answerable"] is not None]
    result = refusal_metrics([(s["answerable"], s["answered"]) for s in ok])
    unanswerable = [s for s in ok if s["answerable"] is False]
    faith = [judge[(s["id"], "faithfulness")]["faithfulness"] for s in ok
             if (s["id"], "faithfulness") in judge and judge[(s["id"], "faithfulness")].get("faithfulness") is not None]
    result["fabrication_rate_unanswerable"] = _rate(sum(1 for s in unanswerable if s["forbidden_hits"]), len(unanswerable))
    result["faithfulness_mean"] = mean(faith)
    result["by_suite"] = {
        suite: refusal_metrics([(s["answerable"], s["answered"]) for s in ok if s["suite"] == suite])
        for suite in sorted({s["suite"] for s in ok})
    }
    return result


# ---------------------------------------------------------------------------
# 5. Performance
# ---------------------------------------------------------------------------

def _bucket(count: int) -> str:
    return "3+" if count >= 3 else str(count)


def performance_metrics(run_dir: Path, main_records: list[dict[str, Any]]) -> dict[str, Any]:
    turns = []
    for record in main_records:
        for turn in record.get("turns") or []:
            if turn.get("status_code") == 200:
                body = turn.get("response") or {}
                usage = body.get("token_usage") or {}
                turns.append({
                    "id": record["id"], "latency_ms": turn["latency_ms"],
                    "server_ms": (turn.get("trace") or {}).get("server_ms"),
                    "tool_bucket": _bucket(len(body.get("tools_used") or [])),
                    "llm_calls": len((turn.get("trace") or {}).get("llm_calls") or []),
                    "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                })
    buckets = ["0", "1", "2", "3+"]
    cfg = settings()
    latency = {b: latency_summary([t["latency_ms"] for t in turns if t["tool_bucket"] == b]) for b in buckets}
    tokens = {
        b: {
            "n": sum(1 for t in turns if t["tool_bucket"] == b),
            "prompt": mean([t["prompt_tokens"] for t in turns if t["tool_bucket"] == b]),
            "completion": mean([t["completion_tokens"] for t in turns if t["tool_bucket"] == b]),
            "total": mean([t["total_tokens"] for t in turns if t["tool_bucket"] == b]),
        }
        for b in buckets
    }
    mean_prompt = mean([t["prompt_tokens"] for t in turns]) or 0
    mean_completion = mean([t["completion_tokens"] for t in turns]) or 0
    cost = None
    if cfg.price_input_per_1m is not None and cfg.price_output_per_1m is not None:
        cost = (mean_prompt * cfg.price_input_per_1m + mean_completion * cfg.price_output_per_1m) / 1_000_000
    preflights = read_json(run_dir / "raw" / "preflight.json", default=[])
    cold = [p["cold_start_ms"] for p in preflights if p.get("cold_start_ms")]

    load_records = read_jsonl(run_dir / "raw" / "load.jsonl")
    done = {r["level"]: r for r in load_records if r.get("type") == "level_done"}
    load = {}
    for level, marker in sorted(done.items()):
        reqs = [r for r in load_records if r.get("type") == "request" and r["level"] == level and r["attempt"] == marker["attempt"]]
        success = [r for r in reqs if r["status_code"] == 200 and not r["timed_out"]]
        span = (max(r["end_ts"] for r in reqs) - min(r["start_ts"] for r in reqs)) if reqs else None
        load[str(level)] = {
            "requests": len(reqs), "success": len(success),
            "error_rate": _rate(sum(1 for r in reqs if r["status_code"] != 200 and not r["timed_out"]), len(reqs)),
            "timeout_rate": _rate(sum(1 for r in reqs if r["timed_out"]), len(reqs)),
            "throughput_rps": safe_div(len(success), span) if span else None,
            "wall_seconds": marker["wall_seconds"],
            "latency": latency_summary([r["latency_ms"] for r in success]),
            "status_codes": dict(Counter(str(r["status_code"]) for r in reqs)),
        }
    return {
        "n_messages": len(turns),
        "latency_overall": latency_summary([t["latency_ms"] for t in turns]),
        "server_latency_overall": latency_summary([t["server_ms"] for t in turns if t["server_ms"]]),
        "latency_by_tools": latency,
        "tokens_by_tools": tokens,
        "tokens_mean": {"prompt": mean_prompt, "completion": mean_completion, "total": mean([t["total_tokens"] for t in turns])},
        "cost_per_question": cost, "currency": cfg.currency,
        "price_input_per_1m": cfg.price_input_per_1m, "price_output_per_1m": cfg.price_output_per_1m,
        "cold_start_ms": cold[0] if cold else None,
        "load": load,
        "latencies_by_bucket_raw": {b: [t["latency_ms"] for t in turns if t["tool_bucket"] == b] for b in buckets},
    }


# ---------------------------------------------------------------------------
# 6. Robustness
# ---------------------------------------------------------------------------

def robustness_metrics(scored: list[dict[str, Any]], consistency_scored: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [s for s in scored if s["status"] == "ok"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in ok:
        if s.get("paraphrase_group"):
            groups[s["paraphrase_group"]].append(s)
    group_rows = []
    for name, members in sorted(groups.items()):
        outcomes = [m["item_pass"] for m in members if m["item_pass"] is not None]
        group_rows.append({"group": name, "n": len(members), "passed": sum(1 for o in outcomes if o),
                           "consistent": len(set(outcomes)) <= 1 if outcomes else None,
                           "all_pass": all(outcomes) if outcomes else None})
    rated = [g for g in group_rows if g["consistent"] is not None]

    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in consistency_scored:
        if s["status"] == "ok":
            runs[s["id"]].append(s)
    run_rows = []
    for item_id, attempts in sorted(runs.items()):
        outcomes = [a["item_pass"] for a in attempts if a["item_pass"] is not None]
        run_rows.append({
            "id": item_id, "runs": len(attempts), "passed": sum(1 for o in outcomes if o),
            "same_outcome": len(set(outcomes)) <= 1 if outcomes else None,
            "same_answered": len({a["answered"] for a in attempts}) == 1,
            "same_tools": len({tuple(sorted(set(a["tools_used"]))) for a in attempts}) == 1,
        })
    run_rated = [r for r in run_rows if r["same_outcome"] is not None]

    multi = [s for s in ok if s["multi_turn"] and s["item_pass"] is not None]
    security = [s for s in ok if s["suite"] == "security" and s["attack_success"] is not None]
    return {
        "paraphrase": {"groups": len(rated), "consistent_rate": _rate(sum(g["consistent"] for g in rated), len(rated)),
                       "all_pass_rate": _rate(sum(bool(g["all_pass"]) for g in rated), len(rated)), "rows": group_rows},
        "run_to_run": {"items": len(run_rated), "repeats": max((r["runs"] for r in run_rows), default=0),
                       "same_outcome_rate": _rate(sum(r["same_outcome"] for r in run_rated), len(run_rated)),
                       "same_answered_rate": _rate(sum(r["same_answered"] for r in run_rows), len(run_rows)),
                       "same_tools_rate": _rate(sum(r["same_tools"] for r in run_rows), len(run_rows)),
                       "pass_rate_all_runs": _rate(sum(r["passed"] for r in run_rows), sum(r["runs"] for r in run_rows)),
                       "rows": run_rows},
        "multi_turn": {"n": len(multi), "accuracy": _rate(sum(1 for s in multi if s["item_pass"]), len(multi))},
        "security": {"n": len(security), "attack_success_rate": _rate(sum(1 for s in security if s["attack_success"]), len(security)),
                     "by_category": _group_rate(security, "category", "attack_success"),
                     "skipped": sum(1 for s in scored if s["suite"] == "security" and s["status"] == "skipped")},
    }


# ---------------------------------------------------------------------------
# รวมทุกอย่าง
# ---------------------------------------------------------------------------

def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "--short", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([l for l in git("status", "--porcelain").splitlines() if l.strip()])}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in row.items()})


def item_table(scored: list[dict[str, Any]], semantic: dict[str, Any], judge: dict[tuple[str, str], Any]) -> list[dict[str, Any]]:
    rows = []
    for s in scored:
        rubric = (judge.get((s["id"], "rubric")) or {}).get("scores") or {}
        faith = (judge.get((s["id"], "faithfulness")) or {}).get("faithfulness")
        rows.append({
            "id": s["id"], "suite": s["suite"], "category": s["category"], "language": s["language"],
            "answerable": s["answerable"], "status": s["status"], "result": {True: "ผ่าน", False: "ตก", None: "ไม่นับ"}[s["item_pass"]],
            "fail_reasons": " | ".join(s.get("fail_reasons") or []) or s.get("skip_reason") or s.get("error") or "",
            "question": " / ".join(m for m in s.get("messages") or [] if m), "answer": s.get("answer", ""),
            "answered": s.get("answered"), "expected_tools": s.get("expected_tools"), "tools_used": s.get("tools_used"),
            "tool_exact": s.get("tool_exact"), "tool_lenient": s.get("tool_lenient"),
            "unnecessary_tools": s.get("unnecessary_tools"),
            "args_correct": (s.get("args") or {}).get("all_correct"),
            "facts_passed": s.get("facts_passed"), "facts_total": s.get("facts_total"),
            "forbidden_hits": s.get("forbidden_hits"), "attack_success": s.get("attack_success"),
            "latency_s": round(s["latency_ms"] / 1000, 2) if s.get("latency_ms") else None,
            "llm_calls": s.get("llm_calls"), "total_tokens": s.get("total_tokens"),
            "cosine": (semantic.get(s["id"]) or {}).get("cosine"),
            "bertscore_f1": (semantic.get(s["id"]) or {}).get("bertscore_f1"),
            "judge_correctness": rubric.get("correctness"), "judge_completeness": rubric.get("completeness"),
            "judge_no_fabrication": rubric.get("no_fabrication"), "faithfulness": faith,
        })
    return rows


def failures_markdown(scored: list[dict[str, Any]]) -> str:
    failed = [s for s in scored if s["item_pass"] is False]
    lines = ["# รายการข้อที่ไม่ผ่าน", "", f"ทั้งหมด {len(failed)} ข้อ จาก {len(scored)} ข้อ", ""]
    for s in failed:
        lines.append(f"## `{s['id']}` — {s['suite']} / {s['category']} ({s['language']})")
        lines.append("")
        for n, m in enumerate(s.get("messages") or [], start=1):
            lines.append(f"- **ข้อความที่ {n}:** {m}")
        lines.append("")
        answer = (s.get("answer") or "(ไม่มีคำตอบ)").replace("\n", "\n> ")
        lines.append(f"> {answer}")
        lines.append("")
        lines.append(f"- **สิ่งที่ผิด:** {' | '.join(s.get('fail_reasons') or [s.get('error') or ''])}")
        if s.get("status") == "ok":
            lines.append(f"- **tool ที่ควรเรียก:** {', '.join(s['expected_tools']) or '(ไม่เรียก)'} | "
                         f"**ที่เรียกจริง:** {', '.join(s['tools_used']) or '(ไม่เรียก)'} | answered = {s['answered']}")
        lines.append("")
    return "\n".join(lines)


def compute(run_dir: Path) -> dict[str, Any]:
    items = {i["id"]: i for i in load_items()}
    main_records = list(_latest_by(read_jsonl(run_dir / "raw" / "main.jsonl"), lambda r: r["id"]).values())
    scored = score_records(main_records, items)
    consistency_records = list(_latest_by(read_jsonl(run_dir / "raw" / "consistency.jsonl"),
                                          lambda r: (r["id"], r["repeat"])).values())
    consistency_scored = score_records(consistency_records, items)
    semantic = {r["id"]: r for r in read_jsonl(run_dir / "raw" / "semantic.jsonl")}
    judge = {(r["id"], r["kind"]): r for r in read_jsonl(run_dir / "raw" / "judge.jsonl")}
    kappa = read_json(run_dir / "raw" / "kappa.json")
    retrieval, retrieval_rows = retrieval_metrics(run_dir)
    preflights = read_json(run_dir / "raw" / "preflight.json", default=[])
    info = next((p.get("info") for p in reversed(preflights) if p.get("info")), None)

    metrics = {
        "run": {
            "run_dir": run_dir.name, "dates": sorted({r["today"] for r in main_records}),
            "git": git_info(), "server": info, "items_in_dataset": len(items),
            "items_run": len(scored), "items_ok": sum(1 for s in scored if s["status"] == "ok"),
            "items_error": sum(1 for s in scored if s["status"] == "error"),
            "items_skipped": sum(1 for s in scored if s["status"] == "skipped"),
            "skip_reasons": dict(Counter(s.get("skip_reason") for s in scored if s["status"] == "skipped")),
            "messages_sent": sum(len(r.get("turns") or []) for r in main_records),
            "judge_provider": settings().judge_provider,
        },
        "retrieval": retrieval,
        "tools": tool_metrics(scored, items),
        "answer": answer_metrics(scored, semantic, judge, kappa),
        "refusal": refusal_section(scored, judge),
        "performance": performance_metrics(run_dir, main_records),
        "robustness": robustness_metrics(scored, consistency_scored),
        "proposal_baseline": PROPOSAL_BASELINE,
    }
    tables = run_dir / "tables"
    write_csv(tables / "items.csv", item_table(scored, semantic, judge))
    write_csv(tables / "retrieval.csv", retrieval_rows)
    write_csv(tables / "paraphrase_groups.csv", metrics["robustness"]["paraphrase"]["rows"])
    write_csv(tables / "run_to_run.csv", metrics["robustness"]["run_to_run"]["rows"])
    write_csv(tables / "load_test.csv", [{"concurrency": k, **{x: y for x, y in v.items() if x != "latency"},
                                          **{f"latency_{x}_ms": y for x, y in v["latency"].items()}}
                                         for k, v in metrics["performance"]["load"].items()])
    write_csv(tables / "per_tool.csv", [{"tool": t, **v} for t, v in metrics["tools"]["per_tool"].items()])
    write_csv(tables / "security.csv", [
        {"id": s["id"], "category": s["category"], "attack_success": s.get("attack_success"),
         "question": s.get("question"), "answer": s.get("answer"), "fail_reasons": " | ".join(s.get("fail_reasons") or [])}
        for s in scored if s["suite"] == "security"
    ])
    (run_dir / "failures.md").write_text(failures_markdown(scored), encoding="utf-8")
    slim = {k: v for k, v in metrics.items()}
    write_json(run_dir / "metrics.json", slim)
    return {"metrics": metrics, "scored": scored, "retrieval_rows": retrieval_rows,
            "consistency_scored": consistency_scored, "semantic": semantic, "judge": judge}
