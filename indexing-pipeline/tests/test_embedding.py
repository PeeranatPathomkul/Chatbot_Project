"""
ทดสอบโมดูล embedding — ต้องโหลดโมเดลจริง จึงช้ากว่าเทสต์อื่น

ข้ามชุดนี้ได้ด้วย::

    pytest -m "not slow"
"""

from __future__ import annotations

import pytest

from src import config
from src.embedding import cosine_similarity, get_embedder

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embedder():
    return get_embedder()


def test_เวกเตอร์มี_768_มิติ(embedder) -> None:
    vector = embedder.embed_query("เช็คอินกี่โมง")
    assert len(vector) == config.EMBEDDING_DIMENSION
    assert embedder.dimension == config.EMBEDDING_DIMENSION


def test_เวกเตอร์ถูก_normalize_แล้ว(embedder) -> None:
    """ความยาวต้องเป็น 1 เพื่อให้ dot product = cosine similarity"""
    vector = embedder.embed_query("ห้องพักราคาเท่าไหร่")
    length = sum(x * x for x in vector) ** 0.5
    assert length == pytest.approx(1.0, abs=1e-4)


def test_embed_documents_คืนผลเรียงตรงกับ_input(embedder) -> None:
    texts = ["เช็คอิน 14.00 น.", "สระว่ายน้ำเปิด 07.00 น.", "ห้อง Deluxe 3,800 บาท"]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == len(texts)
    # embed ตัวเดียวซ้ำ ต้องได้เวกเตอร์เดียวกับตอน embed เป็น batch
    single = embedder.embed_documents([texts[1]])[0]
    assert cosine_similarity(vectors[1], single) == pytest.approx(1.0, abs=1e-4)


def test_embed_รายการว่างคืน_list_ว่าง(embedder) -> None:
    assert embedder.embed_documents([]) == []
    assert embedder.embed_queries([]) == []


def test_prefix_ของ_e5_ถูกใส่คนละแบบระหว่าง_query_กับ_document(embedder) -> None:
    """query กับ passage ของข้อความเดียวกันต้องไม่ได้เวกเตอร์เท่ากัน มิฉะนั้นแปลว่าลืมใส่ prefix"""
    text = "ยกเลิกการจองได้ไหม"
    as_query = embedder.embed_query(text)
    as_passage = embedder.embed_documents([text])[0]
    assert cosine_similarity(as_query, as_passage) < 0.999


def test_เอกสารที่ตรงคำถามต้องได้คะแนนสูงกว่าเอกสารที่ไม่เกี่ยว(embedder) -> None:
    """หัวใจของการพิสูจน์ว่า embedding ภาษาไทยใช้งานได้จริง"""
    question = embedder.embed_query("ห้อง Deluxe มีเตียงเสริมไหม")
    relevant, irrelevant = embedder.embed_documents(
        [
            "ห้อง Deluxe สามารถขอเตียงเสริมได้ 1 เตียง คิดค่าบริการคืนละ 800 บาท",
            "สระว่ายน้ำระบบเกลือขนาด 25 เมตร เปิดให้บริการเวลา 07.00 ถึง 20.00 น.",
        ]
    )
    score_relevant = cosine_similarity(question, relevant)
    score_irrelevant = cosine_similarity(question, irrelevant)
    assert score_relevant > score_irrelevant, (
        f"เอกสารที่เกี่ยวข้องได้ {score_relevant:.4f} "
        f"แต่เอกสารที่ไม่เกี่ยวได้ {score_irrelevant:.4f}"
    )


def test_ภาษาไทยกับภาษาอังกฤษความหมายเดียวกันต้องใกล้กัน(embedder) -> None:
    """multilingual-e5 ควรจับได้ว่าไทยกับอังกฤษคู่นี้พูดเรื่องเดียวกัน

    วัดแบบเทียบกัน ไม่ใช้เลขเกณฑ์ตายตัว เพราะคะแนนดิบของ e5 ขึ้นกับความยาว
    และรูปประโยคมาก การตั้งเลขคงที่จะกลายเป็นเทสต์เปราะที่ไม่ได้วัดอะไรจริง
    """
    thai, same_meaning, different_meaning = embedder.embed_documents(
        [
            "เวลาเช็คอินคือ 14.00 น.",
            "Check-in time is 2 PM.",
            "The swimming pool has a salt water system.",
        ]
    )
    assert cosine_similarity(thai, same_meaning) > cosine_similarity(thai, different_meaning)


def test_นับ_token_ได้และ_chunk_size_ที่ตั้งไว้ไม่เกินเพดานโมเดล(embedder) -> None:
    text = "ห้องพัก" * (config.CHUNK_SIZE // 7)  # ข้อความไทยยาวเท่า CHUNK_SIZE โดยประมาณ
    tokens = embedder.count_tokens(config.PASSAGE_PREFIX + text)
    assert tokens > 0
    assert tokens <= embedder.max_seq_length, (
        f"CHUNK_SIZE={config.CHUNK_SIZE} ตัวอักษร กลายเป็น {tokens} token "
        f"ซึ่งเกินเพดาน {embedder.max_seq_length} ของโมเดล"
    )


def test_get_embedder_คืนตัวเดิมเสมอ() -> None:
    assert get_embedder() is get_embedder()
