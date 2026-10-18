"""metric ทุกตัวทดสอบกับตัวอย่างที่คำนวณด้วยมือได้ (ค่าคาดหวังเขียนวิธีคิดไว้ในคอมเมนต์)"""

import math

import pytest

from harness.metrics import (
    bertscore_from_similarity, cohen_kappa, cosine, hit_at_k, latency_summary, ndcg_at_k, percent_agreement,
    percentile, precision_recall_f1, reciprocal_rank, refusal_metrics, weighted_kappa,
)


def test_hit_at_k():
    relevance = [False, False, True, False, False]
    assert hit_at_k(relevance, 1) == 0.0
    assert hit_at_k(relevance, 3) == 1.0
    assert hit_at_k([False] * 5, 5) == 0.0


def test_mrr_by_hand():
    # คำถาม 3 ข้อ: เจอที่อันดับ 1, อันดับ 3, ไม่เจอ -> (1 + 1/3 + 0) / 3 = 0.4444
    ranks = [[True, False], [False, False, True], [False, False, False]]
    mrr = sum(reciprocal_rank(r) for r in ranks) / len(ranks)
    assert mrr == pytest.approx(4 / 9)
    assert reciprocal_rank([False, True], k=1) == 0.0  # อยู่นอก top-k


def test_ndcg_by_hand():
    # relevant ที่อันดับ 1 และ 3 จาก 2 ตัวในคลัง
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5 ; IDCG = 1 + 1/log2(3) = 1.6309
    value = ndcg_at_k([True, False, True, False, False], 5, total_relevant=2)
    assert value == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([True, False, False], 3, total_relevant=1) == 1.0
    assert ndcg_at_k([False, False], 2, total_relevant=0) == 0.0
    # มี relevant ในคลัง 10 ตัว แต่ IDCG จำกัดที่ k=2: เจอครบ 2 อันดับแรก = 1.0
    assert ndcg_at_k([True, True], 2, total_relevant=10) == 1.0


def test_precision_recall_f1():
    # TP 8, FP 2, FN 4 -> P 0.8, R 0.6667, F1 = 2PR/(P+R) = 0.7273
    r = precision_recall_f1(8, 2, 4)
    assert r["precision"] == pytest.approx(0.8)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 * 0.8 * (2 / 3) / (0.8 + 2 / 3))
    assert precision_recall_f1(0, 0, 3)["precision"] is None


def test_percentile_linear_interpolation():
    data = [10, 20, 30, 40]
    # ตำแหน่ง (n-1) * q = 3 * 0.5 = 1.5 -> ระหว่าง 20 กับ 30 = 25
    assert percentile(data, 50) == 25
    # 3 * 0.95 = 2.85 -> 30 + 0.85 * 10 = 38.5
    assert percentile(data, 95) == pytest.approx(38.5)
    assert percentile([7], 99) == 7
    assert percentile([], 50) is None
    assert latency_summary([1, 2, 3])["mean"] == 2


def test_cohen_kappa_by_hand():
    # ตาราง 2x2: ตรงกัน yes-yes 20, no-no 15, ไม่ตรง 5 + 10, n = 50
    # p_o = 35/50 = 0.7 ; A yes 25/50, B yes 30/50 -> p_e = 0.5*0.6 + 0.5*0.4 = 0.5 ; kappa = (0.7-0.5)/0.5 = 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert percent_agreement(a, b) == pytest.approx(0.7)
    assert cohen_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_weighted_kappa():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # ผิดห่าง 1 ขั้นต้องได้ kappa ถ่วงน้ำหนักสูงกว่าผิดห่าง 4 ขั้น
    near = weighted_kappa([1, 2, 3, 4, 5, 5], [2, 2, 3, 4, 5, 4], [1, 2, 3, 4, 5])
    far = weighted_kappa([1, 2, 3, 4, 5, 5], [5, 2, 3, 4, 5, 1], [1, 2, 3, 4, 5])
    assert near > far


def test_refusal_metrics_by_hand():
    # ตอบไม่ได้ 4 ข้อ: ปฏิเสธ 3 ตอบ 1 ; ตอบได้ 6 ข้อ: ตอบ 5 ปฏิเสธ 1
    rows = [(False, False)] * 3 + [(False, True)] + [(True, True)] * 5 + [(True, False)]
    r = refusal_metrics(rows)
    assert r["precision"] == pytest.approx(3 / 4)  # ปฏิเสธ 4 ครั้ง ถูก 3
    assert r["recall"] == pytest.approx(3 / 4)     # ควรปฏิเสธ 4 ปฏิเสธ 3
    assert r["hallucination_rate"] == pytest.approx(1 / 4)
    assert r["false_refusal_rate"] == pytest.approx(1 / 6)


def test_cosine_and_bertscore():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    # คำตอบ 2 token, อ้างอิง 3 token
    sim = [[0.9, 0.1, 0.2],
           [0.3, 0.8, 0.4]]
    r = bertscore_from_similarity(sim)
    assert r["precision"] == pytest.approx((0.9 + 0.8) / 2)        # max ของแต่ละแถว
    assert r["recall"] == pytest.approx((0.9 + 0.8 + 0.4) / 3)     # max ของแต่ละคอลัมน์
    assert r["f1"] == pytest.approx(2 * r["precision"] * r["recall"] / (r["precision"] + r["recall"]))
