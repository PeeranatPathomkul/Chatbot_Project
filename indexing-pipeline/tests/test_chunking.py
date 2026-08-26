"""ทดสอบการแบ่ง chunk ภาษาไทย (ไม่ต้องโหลดโมเดล จึงรันเร็ว)"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.chunking import (
    Chunk,
    chunk_directory,
    chunk_document,
    ends_mid_word,
    find_broken_boundaries,
    load_document,
    load_documents,
    parse_front_matter,
    split_into_sections,
    starts_mid_word,
)


# --------------------------------------------------------------------------
# front matter และการหาหมวดหมู่
# --------------------------------------------------------------------------


def test_parse_front_matter_แยก_metadata_ออกจากเนื้อหา() -> None:
    raw = "---\ncategory: นโยบาย\ntitle: การยกเลิก\n---\nเนื้อหาจริง"
    meta, body = parse_front_matter(raw)
    assert meta == {"category": "นโยบาย", "title": "การยกเลิก"}
    assert body == "เนื้อหาจริง"


def test_parse_front_matter_ไฟล์ที่ไม่มี_front_matter() -> None:
    raw = "เนื้อหาล้วน ๆ ไม่มี front matter"
    meta, body = parse_front_matter(raw)
    assert meta == {}
    assert body == raw


def test_หมวดหมู่มาจาก_front_matter_ก่อน(thai_document) -> None:
    assert thai_document.category == "ห้องพัก"


def test_หมวดหมู่_fallback_ไปที่ชื่อไฟล์(tmp_path: Path) -> None:
    path = tmp_path / "faq.txt"
    path.write_text("ถาม: เช็คอินกี่โมง\nตอบ: 14.00 น.", encoding="utf-8")
    assert load_document(path).category == "FAQ"


def test_หมวดหมู่_fallback_ไปค่าเริ่มต้น(tmp_path: Path) -> None:
    path = tmp_path / "ไฟล์ไม่รู้จัก.txt"
    path.write_text("ข้อความทั่วไป", encoding="utf-8")
    assert load_document(path).category == config.DEFAULT_CATEGORY


# --------------------------------------------------------------------------
# การแบ่งตามหัวข้อ
# --------------------------------------------------------------------------


def test_split_into_sections_แยกตามหัวข้อ_markdown() -> None:
    body = "# ใหญ่\nนำ\n## ก\nเนื้อหา ก\n## ข\nเนื้อหา ข"
    sections = split_into_sections(body)
    headings = [heading for heading, _ in sections]
    assert headings == ["ใหญ่", "ก", "ข"]
    assert sections[1][1] == "เนื้อหา ก"


def test_split_into_sections_ไฟล์ที่ไม่มีหัวข้อ() -> None:
    body = "ข้อความล้วนไม่มีหัวข้อ"
    assert split_into_sections(body) == [("", body)]


def test_chunk_ไม่คร่อมสองหัวข้อ(thai_document) -> None:
    """chunk ที่มาจากหัวข้อ Deluxe ต้องไม่มีเนื้อหาของ Superior ปนมา"""
    chunks = chunk_document(thai_document)
    for chunk in chunks:
        if chunk.heading == "ห้อง Deluxe":
            assert "ซูพีเรีย" not in chunk.text


# --------------------------------------------------------------------------
# metadata ของ chunk
# --------------------------------------------------------------------------


def test_ทุก_chunk_มี_metadata_ครบ(thai_document) -> None:
    for chunk in chunk_document(thai_document):
        metadata = chunk.to_metadata()
        assert metadata["chunk_id"]
        assert metadata["source"] == "rooms.md"
        assert metadata["category"] == "ห้องพัก"
        assert isinstance(metadata["chunk_index"], int)


def test_chunk_id_เป็น_deterministic(thai_document) -> None:
    """แบ่งซ้ำสองรอบต้องได้ ID ชุดเดิมเป๊ะ — เป็นฐานของการ re-index ที่ไม่ซ้ำซ้อน"""
    first = [c.chunk_id for c in chunk_document(thai_document)]
    second = [c.chunk_id for c in chunk_document(thai_document)]
    assert first == second
    assert len(set(first)) == len(first), "chunk_id ต้องไม่ซ้ำกันภายในไฟล์เดียว"


def test_chunk_id_ไม่ชนกันข้ามไฟล์(data_dir: Path) -> None:
    ids = [c.chunk_id for c in chunk_directory(data_dir)]
    assert len(set(ids)) == len(ids)


# --------------------------------------------------------------------------
# หัวใจ: ต้องไม่ตัดกลางคำไทย
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["ิดตาม", "้องพัก", "ำนวน", "ะเบียง"],
)
def test_starts_mid_word_ตรวจจับสระ_วรรณยุกต์ต้นข้อความ(text: str) -> None:
    assert starts_mid_word(text)


@pytest.mark.parametrize("text", ["ห้องพัก", "เตียงเสริม", "การจอง", "Deluxe"])
def test_starts_mid_word_ข้อความปกติต้องไม่ติด(text: str) -> None:
    assert not starts_mid_word(text)


@pytest.mark.parametrize("text", ["ราคาห้องเ", "กรุณาแจ้งไ"])
def test_ends_mid_word_ตรวจจับสระหน้าค้างท้ายข้อความ(text: str) -> None:
    assert ends_mid_word(text)


def test_เอกสารตัวอย่างทั้งหมดไม่มี_chunk_ที่ตัดกลางคำ(data_dir: Path) -> None:
    """เทสต์สำคัญที่สุดของโมดูลนี้ — พิสูจน์ว่า separators ที่เลือกใช้ได้จริงกับไทย"""
    chunks = chunk_directory(data_dir)
    problems = find_broken_boundaries(chunks)
    assert not problems, f"พบ chunk ที่ตัดกลางคำ: {problems[:5]}"


def test_ข้อความไทยยาวไม่มีเว้นวรรคเลย_ยังแบ่งได้() -> None:
    """เคสสุดโต่ง: ไทยล้วนไม่มีเว้นวรรค ต้อง fallback ไปตัดตามตัวอักษรได้ ไม่ crash"""
    from src.chunking import SourceDocument

    body = "ห้องพักของเรามีหลายประเภทให้เลือก" * 40
    document = SourceDocument(path=Path("dense.txt"), body=body)
    chunks = chunk_document(document)
    assert len(chunks) > 1
    assert all(len(c.text) <= config.CHUNK_SIZE + 200 for c in chunks)


# --------------------------------------------------------------------------
# ขนาด chunk
# --------------------------------------------------------------------------


def test_chunk_ไม่ยาวเกินที่กำหนดมากเกินไป(data_dir: Path) -> None:
    """เผื่อความยาวของบรรทัดบริบทที่เติมเข้าไป จึงอนุญาตให้เกินได้เล็กน้อย"""
    for chunk in chunk_directory(data_dir):
        assert len(chunk.text) <= config.CHUNK_SIZE + 200


def test_ปรับ_chunk_size_ผ่านพารามิเตอร์ได้(thai_document) -> None:
    small = chunk_document(thai_document, chunk_size=150, chunk_overlap=20)
    large = chunk_document(thai_document, chunk_size=2000, chunk_overlap=0)
    assert len(small) > len(large)


def test_ทิ้ง_chunk_ที่สั้นเกินไป(tmp_path: Path) -> None:
    path = tmp_path / "short.md"
    path.write_text("# หัวข้อ\n\nสั้นมาก\n", encoding="utf-8")
    assert chunk_document(load_document(path)) == []


# --------------------------------------------------------------------------
# การอ่านโฟลเดอร์
# --------------------------------------------------------------------------


def test_อ่านเอกสารตัวอย่างครบทุกไฟล์(data_dir: Path) -> None:
    documents = load_documents(data_dir)
    sources = {d.source for d in documents}
    assert {"room_types.md", "booking_policy.md", "faq.md"} <= sources


def test_โฟลเดอร์ที่ไม่มีอยู่จริงต้องโยน_FileNotFoundError(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "ไม่มีจริง")


def test_บรรทัดบริบทถูกเติมไว้หัว_chunk(thai_document) -> None:
    if not config.PREPEND_HEADING_CONTEXT:
        pytest.skip("ปิด PREPEND_HEADING_CONTEXT อยู่")
    chunk: Chunk = chunk_document(thai_document)[0]
    assert chunk.text.startswith("[หมวด: ห้องพัก")
