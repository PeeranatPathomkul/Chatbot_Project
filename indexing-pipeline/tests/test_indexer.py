"""
ทดสอบ indexer แบบ end-to-end กับ ChromaDB จริง (เขียนลงโฟลเดอร์ชั่วคราว)

ต้องโหลดโมเดล embedding จึงจัดเป็นเทสต์ช้า ข้ามได้ด้วย ``pytest -m "not slow"``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.indexer import (
    collection_stats,
    delete_chunks_of_source,
    get_client,
    get_collection,
    index_directory,
    search,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def indexed(tmp_path_factory) -> tuple[Path, object]:
    """index เอกสารตัวอย่างลง ChromaDB ในโฟลเดอร์ชั่วคราว (ทำครั้งเดียวต่อ module)"""
    persist_dir = tmp_path_factory.mktemp("chroma_test")
    stats = index_directory(persist_dir=persist_dir, reset=True, show_progress=False)
    collection = get_collection(get_client(persist_dir))
    return persist_dir, (stats, collection)


def test_index_อ่านครบทุกไฟล์และสร้าง_chunk_ได้(indexed) -> None:
    _, (stats, collection) = indexed
    assert stats.files_read == 5
    assert stats.chunks_created > 20
    assert collection.count() == stats.chunks_created
    assert stats.elapsed_seconds > 0


def test_ไม่มี_chunk_ที่ตัดกลางคำไทยตอน_index(indexed) -> None:
    _, (stats, _) = indexed
    assert stats.broken_boundaries == []


def test_collection_ใช้_cosine_metric(indexed) -> None:
    """สำคัญ: ถ้าเป็น l2 อันดับผลลัพธ์ตอน retrieval จะไม่ตรงกับที่ออกแบบไว้"""
    _, (_, collection) = indexed
    assert collection.metadata["hnsw:space"] == "cosine"


def test_metadata_ถูกเก็บลง_chroma_ครบ(indexed) -> None:
    _, (_, collection) = indexed
    payload = collection.get(limit=5, include=["metadatas", "documents"])
    for metadata in payload["metadatas"]:
        assert metadata["source"].endswith((".md", ".txt"))
        assert metadata["category"]
        assert metadata["chunk_id"]
        assert isinstance(metadata["chunk_index"], int)


def test_รัน_index_ซ้ำแล้วจำนวนระเบียนไม่บานปลาย(indexed) -> None:
    """หัวใจของการ re-index: deterministic ID + ลบตาม source ก่อนเขียน"""
    persist_dir, (stats, collection) = indexed
    before = collection.count()

    index_directory(persist_dir=persist_dir, reset=False, show_progress=False)

    after = get_collection(get_client(persist_dir)).count()
    assert after == before, f"รันซ้ำแล้วระเบียนเปลี่ยนจาก {before} เป็น {after}"


def test_ไฟล์ที่สั้นลงต้องไม่ทิ้ง_chunk_เก่าค้างไว้(tmp_path: Path) -> None:
    """เคสที่ deterministic ID อย่างเดียวแก้ไม่ได้ ต้องพึ่ง delete-by-source"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    persist_dir = tmp_path / "chroma"

    long_text = (
        "---\ncategory: FAQ\n---\n\n"
        + "\n\n".join(
            f"ถาม: คำถามข้อที่ {i} เกี่ยวกับการเข้าพักที่รีสอร์ทของเรา\n"
            f"ตอบ: คำตอบข้อที่ {i} อธิบายรายละเอียดเงื่อนไขการเข้าพักอย่างครบถ้วนชัดเจน"
            for i in range(12)
        )
    )
    target = data_dir / "faq.md"
    target.write_text(long_text, encoding="utf-8")

    first = index_directory(data_dir=data_dir, persist_dir=persist_dir, reset=True,
                            show_progress=False)
    assert first.chunks_created > 3

    # ตัดเนื้อหาให้เหลือน้อยลงมาก แล้ว index ใหม่
    target.write_text(
        "---\ncategory: FAQ\n---\n\nถาม: เช็คอินกี่โมง\nตอบ: เช็คอินได้ตั้งแต่ 14.00 น. เป็นต้นไป",
        encoding="utf-8",
    )
    second = index_directory(data_dir=data_dir, persist_dir=persist_dir, reset=False,
                             show_progress=False)

    collection = get_collection(get_client(persist_dir))
    assert collection.count() == second.chunks_created
    assert collection.count() < first.chunks_created
    assert second.chunks_deleted == first.chunks_created


def test_delete_chunks_of_source_ลบเฉพาะไฟล์ที่ระบุ(indexed) -> None:
    persist_dir, _ = indexed
    # ใช้ collection แยกชื่อ เพื่อไม่กระทบเทสต์อื่นที่ใช้ fixture เดียวกัน
    client = get_client(persist_dir)
    collection = client.get_or_create_collection(
        name="tmp_delete_test", metadata=config.COLLECTION_METADATA
    )
    collection.upsert(
        ids=["a::0000", "b::0000"],
        embeddings=[[0.1] * config.EMBEDDING_DIMENSION, [0.2] * config.EMBEDDING_DIMENSION],
        documents=["ข้อความ ก", "ข้อความ ข"],
        metadatas=[{"source": "a.md"}, {"source": "b.md"}],
    )
    removed = delete_chunks_of_source(collection, "a.md")
    assert removed == 1
    assert collection.count() == 1
    client.delete_collection("tmp_delete_test")


# --------------------------------------------------------------------------
# การค้นคืนภาษาไทย
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected_sources"),
    [
        ("ห้อง Deluxe มีเตียงเสริมไหม", {"room_types.md", "faq.md"}),
        ("เช็คอินกี่โมง", {"booking_policy.md", "faq.md"}),
        ("ยกเลิกการจองได้ไหม", {"booking_policy.md", "faq.md"}),
        ("สระว่ายน้ำเปิดกี่โมง", {"facilities.md"}),
        ("ห้องพูลวิลล่าราคาเท่าไหร่", {"rates_and_packages.md"}),
    ],
)
def test_query_ภาษาไทยดึงไฟล์ที่ถูกต้องติดมาใน_top_k(
    indexed, question: str, expected_sources: set
) -> None:
    """เกณฑ์คือ "ไฟล์ที่ตอบคำถามได้ต้องติดมาใน top-k" ไม่ใช่ "ต้องเป็นอันดับ 1"

    เพราะขั้นตอน generation จะส่ง chunk ทั้ง k ชิ้นให้ LLM อ่าน ขอแค่ chunk
    ที่มีคำตอบติดมาในชุดนั้น LLM ก็ตอบได้ถูก การบังคับ top-1 เป็นเกณฑ์ที่เข้มเกิน
    ความเป็นจริงของ RAG และทำให้เทสต์ล้มจากการสลับอันดับเพียงเล็กน้อย
    """
    _, (_, collection) = indexed
    top_k = 3
    hits = search(question, top_k=top_k, collection=collection)

    assert hits, "ไม่ได้ผลลัพธ์เลย"

    retrieved_sources = [hit["metadata"]["source"] for hit in hits]
    assert expected_sources & set(retrieved_sources), (
        f"top-{top_k} ได้ {retrieved_sources} แต่คาดหวังให้มี {expected_sources} "
        f"ติดมาด้วย (score อันดับ 1 = {hits[0]['similarity']:.4f})"
    )
    assert hits[0]["similarity"] >= 0.80


def test_ผลลัพธ์เรียงจากคะแนนมากไปน้อย(indexed) -> None:
    _, (_, collection) = indexed
    hits = search("เช็คอินกี่โมง", top_k=5, collection=collection)
    scores = [h["similarity"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_similarity_แปลงจาก_distance_ถูกต้อง(indexed) -> None:
    _, (_, collection) = indexed
    for hit in search("ยกเลิกการจองได้ไหม", top_k=3, collection=collection):
        assert hit["similarity"] == pytest.approx(1.0 - hit["distance"])
        assert 0.0 <= hit["similarity"] <= 1.0


def test_กรองตามหมวดหมู่ได้(indexed) -> None:
    _, (_, collection) = indexed
    hits = search("เช็คอินกี่โมง", top_k=3, category="FAQ", collection=collection)
    assert hits
    assert all(hit["metadata"]["category"] == "FAQ" for hit in hits)


def test_คำถามที่ไม่เกี่ยวข้องได้คะแนนต่ำกว่าคำถามที่เกี่ยวข้อง(indexed) -> None:
    _, (_, collection) = indexed
    relevant = search("เช็คอินกี่โมง", top_k=1, collection=collection)[0]
    unrelated = search("วิธีเปลี่ยนถ่ายน้ำมันเครื่องรถยนต์", top_k=1, collection=collection)[0]
    assert relevant["similarity"] > unrelated["similarity"]


def test_collection_stats_สรุปถูกต้อง(indexed) -> None:
    _, (stats, collection) = indexed
    summary = collection_stats(collection)
    assert summary["count"] == stats.chunks_created
    assert summary["metric"] == "cosine"
    assert sum(summary["by_source"].values()) == stats.chunks_created
    assert "FAQ" in summary["by_category"]
