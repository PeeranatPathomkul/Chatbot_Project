"""
ฟังก์ชันคำนวณ metric แบบบริสุทธิ์ (ไม่มี I/O) — ทุกตัวมี pytest ใน evaluate/tests/test_metrics.py

นิยามตามเอกสารต้นทาง:
- MRR: ค่าเฉลี่ยของ 1/rank ของผลลัพธ์ที่เกี่ยวข้องตัวแรก (0 ถ้าไม่พบใน top-k)
- nDCG@k (Järvelin & Kekäläinen, 2002): DCG = Σ rel_i / log2(i + 1), i เริ่มที่ 1
  IDCG = DCG ของการเรียงที่ดีที่สุด (relevant ทั้งหมดอยู่บนสุด จำกัดที่ k)
- Cohen's kappa: (p_o - p_e) / (1 - p_e) และแบบถ่วงน้ำหนักกำลังสอง (quadratic weighted)
- percentile: interpolation แบบเชิงเส้น (วิธีเดียวกับค่าเริ่มต้นของ numpy.percentile)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Hashable, Sequence


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool], k: int | None = None) -> float:
    for rank, relevant in enumerate(relevance[:k] if k else relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def dcg(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(relevance: Sequence[bool | float], k: int, total_relevant: int | None = None) -> float:
    """total_relevant = จำนวนเอกสารที่เกี่ยวข้องทั้งหมดในคลัง (ใช้คำนวณ IDCG)

    ไม่ระบุ = นับจากที่อยู่ใน relevance เอง ถ้าไม่มีเอกสารที่เกี่ยวข้องเลยคืน 0
    """
    gains = [float(r) for r in relevance]
    ideal_count = total_relevant if total_relevant is not None else int(sum(1 for g in gains if g > 0))
    ideal = dcg([1.0] * min(ideal_count, k), k)
    return dcg(gains, k) / ideal if ideal > 0 else 0.0


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def percentile(values: Sequence[float], q: float) -> float | None:
    """q ในช่วง 0-100, linear interpolation ระหว่างอันดับ"""
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    if len(data) == 1:
        return float(data[0])
    position = (len(data) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(data[lower])
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


def mean(values: Sequence[float]) -> float | None:
    data = [v for v in values if v is not None]
    return sum(data) / len(data) if data else None


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    data = [v for v in values if v is not None]
    return {
        "n": len(data),
        "mean": mean(data),
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def cohen_kappa(rater_a: Sequence[Hashable], rater_b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa แบบไม่ถ่วงน้ำหนัก"""
    if len(rater_a) != len(rater_b) or not rater_a:
        return None
    n = len(rater_a)
    observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    count_a, count_b = Counter(rater_a), Counter(rater_b)
    expected = sum(count_a[c] * count_b[c] for c in set(count_a) | set(count_b)) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def weighted_kappa(rater_a: Sequence[int], rater_b: Sequence[int], categories: Sequence[int]) -> float | None:
    """Cohen's kappa ถ่วงน้ำหนักกำลังสอง เหมาะกับคะแนนเรียงลำดับ 1-5 (ผิด 1 ขั้นโดนน้อยกว่าผิด 4 ขั้น)"""
    if len(rater_a) != len(rater_b) or not rater_a:
        return None
    cats = list(categories)
    index = {c: i for i, c in enumerate(cats)}
    size, n = len(cats), len(rater_a)
    observed = [[0.0] * size for _ in range(size)]
    for a, b in zip(rater_a, rater_b):
        observed[index[a]][index[b]] += 1
    row = [sum(observed[i]) for i in range(size)]
    col = [sum(observed[i][j] for i in range(size)) for j in range(size)]
    weight = lambda i, j: ((i - j) ** 2) / ((size - 1) ** 2)  # noqa: E731
    num = sum(weight(i, j) * observed[i][j] for i in range(size) for j in range(size))
    den = sum(weight(i, j) * row[i] * col[j] / n for i in range(size) for j in range(size))
    return 1 - num / den if den else None


def percent_agreement(rater_a: Sequence[Hashable], rater_b: Sequence[Hashable]) -> float | None:
    if len(rater_a) != len(rater_b) or not rater_a:
        return None
    return sum(1 for a, b in zip(rater_a, rater_b) if a == b) / len(rater_a)


def refusal_metrics(rows: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """rows = (answerable, answered) ของแต่ละข้อ

    คลาสบวกคือ "ควรปฏิเสธ" (answerable=False) และการทำนายบวกคือ "ปฏิเสธจริง" (answered=False)
    - Hallucination Rate = ตอบทั้งที่ไม่มีข้อมูล / คำถามที่ไม่มีข้อมูลทั้งหมด (= 1 - recall)
    - False Refusal Rate = ปฏิเสธทั้งที่มีข้อมูล / คำถามที่มีข้อมูลทั้งหมด
    """
    tp = sum(1 for answerable, answered in rows if not answerable and not answered)
    fn = sum(1 for answerable, answered in rows if not answerable and answered)
    fp = sum(1 for answerable, answered in rows if answerable and not answered)
    tn = sum(1 for answerable, answered in rows if answerable and answered)
    prf = precision_recall_f1(tp, fp, fn)
    return {
        **prf,
        "tn": tn,
        "confusion": {"unanswerable_refused": tp, "unanswerable_answered": fn,
                      "answerable_refused": fp, "answerable_answered": tn},
        "hallucination_rate": safe_div(fn, tp + fn),
        "false_refusal_rate": safe_div(fp, fp + tn),
        "n": len(rows),
    }


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def bertscore_from_similarity(sim: Sequence[Sequence[float]]) -> dict[str, float]:
    """BERTScore (Zhang et al., 2020) จากเมทริกซ์ cosine ระหว่าง token: แถว = คำตอบ, คอลัมน์ = อ้างอิง

    Precision = เฉลี่ยของ max ในแต่ละแถว, Recall = เฉลี่ยของ max ในแต่ละคอลัมน์ (greedy matching ไม่ถ่วง idf)
    """
    if not sim or not sim[0]:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = sum(max(row) for row in sim) / len(sim)
    recall = sum(max(sim[i][j] for i in range(len(sim))) for j in range(len(sim[0]))) / len(sim[0])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
