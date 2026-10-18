"""
กราฟทุกตัวของรายงาน (matplotlib, PNG) — ทุกกราฟมีชื่อ แกน หน่วย และจำนวนตัวอย่าง (n)

**ฟอนต์ไทย** ฟอนต์ค่าเริ่มต้นของ matplotlib (DejaVu Sans) ไม่มีอักษรไทย ข้อความจะออกมาเป็นกล่องสี่เหลี่ยม
จึงเลือกฟอนต์ที่มีใน Windows ตามลำดับ แล้วตรวจจริงว่าไฟล์ฟอนต์มี glyph ของอักษรไทยครบก่อนใช้
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

THAI_FONT_CANDIDATES = ("Leelawadee UI", "Leelawadee", "Tahoma", "TH Sarabun New", "Noto Sans Thai", "Sarabun")
_THAI_SAMPLE = "ภาษาไทยทดสอบกราฟคำถามที่ตอบได้ไม่ได้"
COLORS = {"th": "#2f6db5", "en": "#e08a2c", "all": "#5a5a5a", "good": "#3a9a5b", "bad": "#c8453b", "neutral": "#7d8fa3"}


def setup_thai_font() -> str:
    for name in THAI_FONT_CANDIDATES:
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
        except ValueError:
            continue
        charmap = FT2Font(path).get_charmap()
        if all(ord(ch) in charmap for ch in set(_THAI_SAMPLE)):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    raise RuntimeError(f"ไม่พบฟอนต์ที่มีอักษรไทย ({', '.join(THAI_FONT_CANDIDATES)}) — กราฟจะเป็นกล่องสี่เหลี่ยม")


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _pct(value: float | None) -> float:
    return 0.0 if value is None else value * 100


def _label_bars(ax, bars, fmt="{:.0f}%") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height), (bar.get_x() + bar.get_width() / 2, height), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points")


def _heatmap(ax, matrix: list[list[float]], rows: list[str], cols: list[str], cmap="Blues") -> None:
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    peak = max((max(r) for r in matrix), default=0) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:g}", ha="center", va="center", fontsize=8,
                    color="white" if value > peak * 0.6 else "black")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def make_all(result: dict[str, Any], charts_dir: Path) -> dict[str, Path]:
    font = setup_thai_font()
    charts_dir.mkdir(parents=True, exist_ok=True)
    m = result["metrics"]
    out: dict[str, Path] = {}

    # ---------- 1. Retrieval ----------
    retrieval = m.get("retrieval") or {}
    if retrieval:
        golden = retrieval["golden"]
        names = ["Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5"]
        keys = ["hit1", "hit3", "hit5", "rr", "ndcg5"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        width = 0.26
        for offset, (label, data, color) in enumerate([
            (f"ไทย (n={golden['by_language']['th']['n']})", golden["by_language"]["th"], COLORS["th"]),
            (f"อังกฤษ (n={golden['by_language']['en']['n']})", golden["by_language"]["en"], COLORS["en"]),
            (f"รวม (n={golden['n']})", golden, COLORS["all"]),
        ]):
            bars = ax.bar([i + (offset - 1) * width for i in range(len(keys))], [_pct(data.get(k)) for k in keys], width,
                          label=label, color=color)
            _label_bars(ax, bars)
        ax.set_xticks(range(len(keys)), names)
        ax.set_ylabel("คะแนน (%)")
        ax.set_ylim(0, 112)
        ax.set_title(f"ความแม่นยำการค้นคืน (ระดับ chunk) แยกภาษา — top_k={retrieval['top_k']}, {retrieval['chunk_count']} chunk")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        out["retrieval_hitk"] = _save(fig, charts_dir / "retrieval_hitk.png")

        ans, unans = retrieval["answerable_scores"], retrieval["unanswerable_scores"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})
        low = int(min(ans + unans, default=0.7) * 100) - 1
        bins = [x / 100 for x in range(low, 101)]
        ax1.hist(ans, bins=bins, alpha=0.65, label=f"ตอบได้ (n={len(ans)})", color=COLORS["good"])
        ax1.hist(unans, bins=bins, alpha=0.65, label=f"ตอบไม่ได้ (n={len(unans)})", color=COLORS["bad"])
        threshold = retrieval["score_distribution"]["best_threshold"]
        if threshold["threshold"] is not None:
            ax1.axvline(threshold["threshold"], color="black", linestyle="--", linewidth=1,
                        label=f"threshold ที่ดีที่สุด {threshold['threshold']:.3f} (ถูก {threshold['accuracy'] * 100:.0f}%)")
        ax1.set_xlabel("cosine similarity ของ chunk อันดับ 1")
        ax1.set_ylabel("จำนวนคำถาม (ข้อ)")
        ax1.set_title("การกระจายคะแนนอันดับ 1: คำถามที่ตอบได้ vs ตอบไม่ได้")
        ax1.legend(fontsize=8)
        ax2.boxplot([ans, unans], tick_labels=["ตอบได้", "ตอบไม่ได้"])
        ax2.set_ylabel("cosine similarity")
        ax2.set_title(f"box plot (n={len(ans)} / {len(unans)})")
        out["retrieval_scores"] = _save(fig, charts_dir / "retrieval_scores.png")

    # ---------- 2. Tool calling ----------
    tools = m["tools"]
    if tools["n"]:
        labels = tools["confusion_labels"]
        matrix = [[tools["confusion"][r][c] for c in labels] for r in labels]
        fig, ax = plt.subplots(figsize=(8.5, 7))
        _heatmap(ax, matrix, labels, labels)
        ax.set_xlabel("tool ที่เรียกจริง")
        ax.set_ylabel("tool ที่ควรเรียก (เฉลย)")
        ax.set_title(f"Confusion matrix ราย tool (ควรเรียก vs เรียกจริง), n={tools['n']} ข้อ\nตัวเลข = จำนวนข้อ")
        out["tool_confusion"] = _save(fig, charts_dir / "tool_confusion.png")

        suites = list(tools["selection_lenient_by_suite"])
        fig, ax = plt.subplots(figsize=(9, 4.2))
        width = 0.38
        exact = [_pct(tools["selection_exact_by_suite"][s]["rate"]) for s in suites]
        lenient = [_pct(tools["selection_lenient_by_suite"][s]["rate"]) for s in suites]
        _label_bars(ax, ax.bar([i - width / 2 for i in range(len(suites))], exact, width, label="ตรงเป๊ะ (exact set)", color=COLORS["th"]))
        _label_bars(ax, ax.bar([i + width / 2 for i in range(len(suites))], lenient, width, label="ครบ + เรียกเกินเฉพาะที่อนุญาต", color=COLORS["good"]))
        ax.set_xticks(range(len(suites)), [f"{s}\n(n={tools['selection_lenient_by_suite'][s]['n']})" for s in suites], fontsize=8)
        ax.set_ylabel("Tool Selection Accuracy (%)")
        ax.set_ylim(0, 112)
        ax.set_title(f"ความถูกต้องในการเลือก tool แยกหมวด (n={tools['n']})")
        ax.legend(fontsize=8)
        out["tool_accuracy"] = _save(fig, charts_dir / "tool_accuracy.png")

        per_tool = {t: v for t, v in tools["per_tool"].items() if (v["tp"] + v["fn"] + v["fp"])}
        fig, ax = plt.subplots(figsize=(9, 4.2))
        names = list(per_tool)
        _label_bars(ax, ax.bar([i - 0.2 for i in range(len(names))], [_pct(per_tool[t]["precision"]) for t in names], 0.4, label="Precision", color=COLORS["th"]))
        _label_bars(ax, ax.bar([i + 0.2 for i in range(len(names))], [_pct(per_tool[t]["recall"]) for t in names], 0.4, label="Recall", color=COLORS["en"]))
        ax.set_xticks(range(len(names)), [f"{t}\n(เฉลย {per_tool[t]['tp'] + per_tool[t]['fn']} ข้อ)" for t in names], fontsize=7, rotation=15)
        ax.set_ylabel("ร้อยละ (%)")
        ax.set_ylim(0, 112)
        ax.set_title("Precision / Recall ราย tool")
        ax.legend(fontsize=8)
        out["tool_precision_recall"] = _save(fig, charts_dir / "tool_precision_recall.png")

    # ---------- 3. Answer correctness ----------
    answer = m["answer"]
    if answer["n_facts"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]})
        by_suite = answer["fact_accuracy_by_suite"]
        bars = ax1.bar(range(len(by_suite)), [_pct(v["rate"]) for v in by_suite.values()], color=COLORS["good"])
        _label_bars(ax1, bars)
        ax1.set_xticks(range(len(by_suite)), [f"{k}\n(n={v['n']})" for k, v in by_suite.items()], fontsize=8)
        ax1.set_ylabel("Key-Fact Accuracy (%)")
        ax1.set_ylim(0, 112)
        ax1.set_title(f"Key-Fact Accuracy แยกหมวด (n={answer['n_facts']} ข้อเท็จจริง)")
        by_lang = answer["fact_accuracy_by_language"]
        bars = ax2.bar(range(len(by_lang)), [_pct(v["rate"]) for v in by_lang.values()],
                       color=[COLORS.get(k, COLORS["all"]) for k in by_lang])
        _label_bars(ax2, bars)
        ax2.set_xticks(range(len(by_lang)), [f"{'ไทย' if k == 'th' else 'อังกฤษ'}\n(n={v['n']})" for k, v in by_lang.items()])
        ax2.set_ylim(0, 112)
        ax2.set_title("แยกภาษา")
        out["keyfact_accuracy"] = _save(fig, charts_dir / "keyfact_accuracy.png")

    judge = answer["judge"]
    if judge["n"]:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
        titles = {"correctness": "ความถูกต้อง", "completeness": "ความครบถ้วน", "no_fabrication": "ไม่แต่งข้อมูล"}
        for ax, dim in zip(axes, ("correctness", "completeness", "no_fabrication")):
            dist = judge["distribution"][dim]
            bars = ax.bar(list(dist), list(dist.values()), color=COLORS["th"])
            _label_bars(ax, bars, "{:.0f}")
            ax.set_xlabel("คะแนน (1-5)")
            ax.set_title(f"{titles[dim]} (เฉลี่ย {judge['means'][dim]:.2f})")
        axes[0].set_ylabel("จำนวนข้อ")
        fig.suptitle(f"การกระจายคะแนน LLM-as-a-Judge ({judge['provider']} / {judge['model']}), n={judge['n']}", fontsize=10)
        out["judge_distribution"] = _save(fig, charts_dir / "judge_distribution.png")

    semantic = answer["semantic"]
    if semantic["n"]:
        by_suite = semantic["by_suite"]
        fig, ax = plt.subplots(figsize=(9, 4))
        names = list(by_suite)
        _label_bars(ax, ax.bar([i - 0.2 for i in range(len(names))], [by_suite[s]["cosine"] for s in names], 0.4,
                               label="Cosine (e5)", color=COLORS["th"]), "{:.2f}")
        _label_bars(ax, ax.bar([i + 0.2 for i in range(len(names))], [by_suite[s]["bertscore_f1"] for s in names], 0.4,
                               label="BERTScore F1", color=COLORS["en"]), "{:.2f}")
        ax.set_xticks(range(len(names)), [f"{s}\n(n={by_suite[s]['n']})" for s in names], fontsize=8)
        ax.set_ylabel("ค่าความคล้าย (0-1)")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"ความคล้ายเชิงความหมายกับคำตอบอ้างอิง แยกหมวด (n={semantic['n']})")
        ax.legend(fontsize=8)
        out["semantic_similarity"] = _save(fig, charts_dir / "semantic_similarity.png")

    # ---------- 4. Refusal ----------
    refusal = m["refusal"]
    if refusal["n"]:
        c = refusal["confusion"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
        _heatmap(ax1, [[c["answerable_answered"], c["answerable_refused"]], [c["unanswerable_answered"], c["unanswerable_refused"]]],
                 ["มีข้อมูล (ตอบได้)", "ไม่มีข้อมูล"], ["ตอบ", "ปฏิเสธ"], cmap="Greens")
        ax1.set_xlabel("ระบบ (answered)")
        ax1.set_ylabel("เฉลย")
        ax1.set_title(f"Confusion matrix การปฏิเสธ (n={refusal['n']})")
        names = ["Refusal\nPrecision", "Refusal\nRecall", "Refusal\nF1", "Hallucination\nRate", "False Refusal\nRate"]
        values = [refusal["precision"], refusal["recall"], refusal["f1"], refusal["hallucination_rate"], refusal["false_refusal_rate"]]
        colors = [COLORS["good"]] * 3 + [COLORS["bad"]] * 2
        bars = ax2.bar(names, [_pct(v) for v in values], color=colors)
        _label_bars(ax2, bars)
        ax2.set_ylabel("ร้อยละ (%)")
        ax2.set_ylim(0, 112)
        ax2.set_title("อัตราต่าง ๆ ของการปฏิเสธ (เขียว = ยิ่งสูงยิ่งดี, แดง = ยิ่งต่ำยิ่งดี)")
        ax2.tick_params(axis="x", labelsize=8)
        out["refusal"] = _save(fig, charts_dir / "refusal.png")

    # ---------- 5. Performance ----------
    perf = m["performance"]
    raw_latency = perf["latencies_by_bucket_raw"]
    buckets = [b for b in ("0", "1", "2", "3+") if raw_latency.get(b)]
    if buckets:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.boxplot([[v / 1000 for v in raw_latency[b]] for b in buckets],
                   tick_labels=[f"{b} tool\n(n={len(raw_latency[b])})" for b in buckets], showmeans=True)
        ax.set_ylabel("เวลาตอบสนอง (วินาที)")
        ax.set_xlabel("จำนวนครั้งที่เรียก tool ในคำถามนั้น")
        ax.set_title(f"เวลาตอบสนองแยกตามจำนวน tool (วัดฝั่ง client, n={perf['n_messages']} ข้อความ)")
        out["latency_by_tools"] = _save(fig, charts_dir / "latency_by_tools.png")

        tokens = perf["tokens_by_tools"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        prompt = [tokens[b]["prompt"] or 0 for b in buckets]
        completion = [tokens[b]["completion"] or 0 for b in buckets]
        ax.bar(buckets, prompt, label="prompt", color=COLORS["th"])
        ax.bar(buckets, completion, bottom=prompt, label="completion", color=COLORS["en"])
        for i, b in enumerate(buckets):
            ax.annotate(f"{prompt[i] + completion[i]:,.0f}", (i, prompt[i] + completion[i]), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(buckets)), [f"{b} tool\n(n={tokens[b]['n']})" for b in buckets])
        ax.set_ylabel("token เฉลี่ยต่อคำถาม")
        ax.set_title("ปริมาณ token เฉลี่ยต่อคำถาม แยกตามจำนวน tool")
        ax.legend(fontsize=8)
        out["tokens_by_tools"] = _save(fig, charts_dir / "tokens_by_tools.png")

    load = perf["load"]
    if load:
        levels = sorted(load, key=int)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        p95 = [(load[l]["latency"]["p95"] or 0) / 1000 for l in levels]
        ax1.plot([int(l) for l in levels], p95, marker="o", color=COLORS["th"], label="latency p95 (วินาที)")
        ax1.set_xlabel("จำนวนคำขอพร้อมกัน (concurrency)")
        ax1.set_ylabel("latency p95 (วินาที)")
        ax1b = ax1.twinx()
        ax1b.plot([int(l) for l in levels], [_pct(load[l]["error_rate"]) + _pct(load[l]["timeout_rate"]) for l in levels],
                  marker="s", color=COLORS["bad"], label="error + timeout (%)")
        ax1b.set_ylabel("error + timeout rate (%)")
        ax1b.set_ylim(0, 105)
        ax1.set_title("latency p95 และอัตราความผิดพลาดตามระดับคำขอพร้อมกัน")
        lines = ax1.get_legend_handles_labels()[0] + ax1b.get_legend_handles_labels()[0]
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
        ax2.bar([str(l) for l in levels], [load[l]["throughput_rps"] or 0 for l in levels], color=COLORS["good"])
        ax2.set_xlabel("concurrency")
        ax2.set_ylabel("throughput (คำขอสำเร็จ/วินาที)")
        ax2.set_title("Throughput (n = " + ", ".join(f"{load[l]['requests']}" for l in levels) + " คำขอ)")
        out["load_test"] = _save(fig, charts_dir / "load_test.png")

    # ---------- 6. Robustness ----------
    rob = m["robustness"]
    names, values, colors, ns = [], [], [], []
    if rob["paraphrase"]["groups"]:
        names.append("Paraphrase\nConsistency"); values.append(rob["paraphrase"]["consistent_rate"]); colors.append(COLORS["good"]); ns.append(f"{rob['paraphrase']['groups']} กลุ่ม")
    if rob["run_to_run"]["items"]:
        names.append(f"Run-to-run\n({rob['run_to_run']['repeats']} ครั้ง)"); values.append(rob["run_to_run"]["same_outcome_rate"]); colors.append(COLORS["good"]); ns.append(f"{rob['run_to_run']['items']} ข้อ")
    if rob["multi_turn"]["n"]:
        names.append("Multi-turn\nAccuracy"); values.append(rob["multi_turn"]["accuracy"]); colors.append(COLORS["good"]); ns.append(f"{rob['multi_turn']['n']} ชุด")
    if rob["security"]["n"]:
        names.append("Attack Success\nRate (เป้า 0%)"); values.append(rob["security"]["attack_success_rate"]); colors.append(COLORS["bad"]); ns.append(f"{rob['security']['n']} ข้อ")
    if names:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        bars = ax.bar([f"{n}\n(n={c})" for n, c in zip(names, ns)], [_pct(v) for v in values], color=colors)
        _label_bars(ax, bars)
        ax.set_ylabel("ร้อยละ (%)")
        ax.set_ylim(0, 112)
        ax.set_title("ความทนทานของระบบ (เขียว = ยิ่งสูงยิ่งดี, แดง = ยิ่งต่ำยิ่งดี)")
        ax.tick_params(axis="x", labelsize=8)
        out["robustness"] = _save(fig, charts_dir / "robustness.png")

    out["_font"] = Path(font)
    return out
