"""
ทดสอบการรองรับหลายภาษาและการตรวจ placeholder ในเอกสารร่าง

ไม่ต้องโหลดโมเดล จึงรันเร็ว
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.chunking import (
    chunk_directory,
    chunk_document,
    find_placeholders,
    load_document,
    load_documents,
    strip_context_header,
)
from src.indexer import _build_where

REAL_DATA_DIR = config.DATA_DIR


@pytest.fixture
def bilingual_dir(tmp_path: Path) -> Path:
    """โฟลเดอร์จำลองที่มีไฟล์ชื่อซ้ำกันคนละภาษา — เคสที่เคยทำให้ chunk_id ชนกัน"""
    thai_body = (
        "---\ncategory: FAQ\n---\n\n# คำถามที่พบบ่อย\n\n"
        "ถาม: เช็คอินได้ตั้งแต่กี่โมง\n"
        "ตอบ: เช็คอินได้ตั้งแต่เวลา 14.00 น. เป็นต้นไป หากมาถึงก่อนเวลาฝากกระเป๋าไว้ได้ฟรี\n"
    )
    english_body = (
        "---\ncategory: FAQ\n---\n\n# Frequently Asked Questions\n\n"
        "Q: What time can I check in?\n"
        "A: Check-in is from 2:00 PM onwards. You may store luggage for free if you arrive early.\n"
    )
    for language, body in (("th", thai_body), ("en", english_body)):
        folder = tmp_path / language
        folder.mkdir()
        (folder / "faq.md").write_text(body, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------
# การแยกภาษา
# --------------------------------------------------------------------------


def test_ภาษาถูกอ่านจากชื่อโฟลเดอร์(bilingual_dir: Path) -> None:
    documents = {d.source: d for d in load_documents(bilingual_dir)}
    assert documents["th/faq.md"].language == "th"
    assert documents["en/faq.md"].language == "en"


def test_front_matter_ชนะชื่อโฟลเดอร์(tmp_path: Path) -> None:
    folder = tmp_path / "th"
    folder.mkdir()
    path = folder / "note.md"
    path.write_text("---\nlanguage: en\n---\n\nEnglish content placed in the th folder.",
                    encoding="utf-8")
    assert load_document(path, base_dir=tmp_path).language == "en"


def test_ไฟล์นอกโฟลเดอร์ภาษาใช้ค่าเริ่มต้น(tmp_path: Path) -> None:
    path = tmp_path / "loose.md"
    path.write_text("เนื้อหาที่ไม่ได้อยู่ในโฟลเดอร์ภาษา", encoding="utf-8")
    assert load_document(path, base_dir=tmp_path).language == config.DEFAULT_LANGUAGE


# --------------------------------------------------------------------------
# chunk_id ต้องไม่ชนกันข้ามภาษา (บั๊กที่เจอตอนเพิ่มภาษาอังกฤษ)
# --------------------------------------------------------------------------


def test_ไฟล์ชื่อซ้ำคนละภาษาต้องไม่ได้_chunk_id_ชนกัน(bilingual_dir: Path) -> None:
    """ถ้า chunk_id ใช้แค่ชื่อไฟล์ ทั้งสองภาษาจะได้ 'faq::0000' เหมือนกัน

    ผลคือตอน upsert ลง Chroma ภาษาหลังจะทับภาษาแรกจนข้อมูลหายไปครึ่งหนึ่ง
    โดยไม่มี error ใด ๆ
    """
    chunks = chunk_directory(bilingual_dir)
    ids = [c.chunk_id for c in chunks]

    assert len(set(ids)) == len(ids), f"chunk_id ซ้ำกัน: {ids}"
    assert any(chunk_id.startswith("th-faq::") for chunk_id in ids)
    assert any(chunk_id.startswith("en-faq::") for chunk_id in ids)


def test_source_เก็บ_path_แบบ_relative(bilingual_dir: Path) -> None:
    sources = {c.source for c in chunk_directory(bilingual_dir)}
    assert sources == {"th/faq.md", "en/faq.md"}


def test_source_ยังเป็นชื่อไฟล์เมื่อไม่ระบุ_base_dir(tmp_path: Path) -> None:
    """ความเข้ากันได้ย้อนหลัง: โฟลเดอร์แบนราบต้องได้ source เป็นชื่อไฟล์เหมือนเดิม"""
    path = tmp_path / "rooms.md"
    path.write_text("# ห้องพัก\n\nรายละเอียดห้องพักที่ยาวพอจะกลายเป็น chunk ได้จริง ๆ",
                    encoding="utf-8")
    assert load_document(path).source == "rooms.md"


def test_language_ถูกเก็บลง_metadata(bilingual_dir: Path) -> None:
    for chunk in chunk_directory(bilingual_dir):
        assert chunk.to_metadata()["language"] in config.SUPPORTED_LANGUAGES


# --------------------------------------------------------------------------
# บรรทัดบริบทต้องเป็นภาษาเดียวกับเนื้อหา
# --------------------------------------------------------------------------


def test_บรรทัดบริบทใช้ภาษาเดียวกับเอกสาร(bilingual_dir: Path) -> None:
    if not config.PREPEND_HEADING_CONTEXT:
        pytest.skip("ปิด PREPEND_HEADING_CONTEXT อยู่")

    by_language = {c.language: c for c in chunk_directory(bilingual_dir)}
    assert by_language["th"].text.startswith("[หมวด:")
    assert by_language["en"].text.startswith("[Category:")


def test_strip_context_header_ตัดได้ทั้งสองภาษา(bilingual_dir: Path) -> None:
    for chunk in chunk_directory(bilingual_dir):
        stripped = strip_context_header(chunk.text)
        assert not stripped.startswith(("[หมวด:", "[Category:"))
        assert stripped


# --------------------------------------------------------------------------
# ตัวกรอง metadata ของ Chroma
# --------------------------------------------------------------------------


def test_build_where_ไม่มีเงื่อนไข() -> None:
    assert _build_where(None, None) is None


def test_build_where_เงื่อนไขเดียวไม่ต้องห่อ_and() -> None:
    assert _build_where("FAQ", None) == {"category": "FAQ"}
    assert _build_where(None, "th") == {"language": "th"}


def test_build_where_สองเงื่อนไขต้องห่อด้วย_and() -> None:
    """Chroma ไม่รับ dict ที่มีหลาย key ตรง ๆ ต้องใช้ $and"""
    assert _build_where("FAQ", "th") == {"$and": [{"category": "FAQ"}, {"language": "th"}]}


# --------------------------------------------------------------------------
# placeholder ในเอกสารร่าง
# --------------------------------------------------------------------------


def test_find_placeholders_เจอ_placeholder(tmp_path: Path) -> None:
    """นับจากเอกสารต้นฉบับ ไม่ใช่จาก chunk เพราะ chunk ตัดบรรทัดพวกนี้ทิ้งไปแล้ว"""
    path = tmp_path / "draft.md"
    path.write_text(
        "# ราคา\n\nห้องดีลักซ์ราคา <<รอเติม: ราคาห้องดีลักซ์>> บาทต่อคืน\n"
        "เช็คอินได้ตั้งแต่ <<รอเติม: เวลาเช็คอิน>> เป็นต้นไปทุกวัน",
        encoding="utf-8",
    )
    found = find_placeholders([load_document(path)])
    texts = {text for _, text in found}
    assert texts == {"<<รอเติม: ราคาห้องดีลักซ์>>", "<<รอเติม: เวลาเช็คอิน>>"}


def test_find_placeholders_เอกสารที่เติมครบแล้วต้องได้ผลว่าง(data_dir: Path) -> None:
    """fixture คือเอกสารที่ 'เติมครบ' แล้ว จึงต้องไม่มี placeholder เหลือ"""
    assert find_placeholders(load_documents(data_dir)) == []


# --------------------------------------------------------------------------
# placeholder ต้องไม่หลุดเข้า chunk ที่ส่งให้ LLM
# --------------------------------------------------------------------------


def test_บรรทัดที่มี_placeholder_ต้องไม่ถูก_index(tmp_path: Path) -> None:
    """วัดจากการทดลองจริงกับ Typhoon: ถ้าปล่อย placeholder ติดไปกับ context
    LLM จะเอาข้อความใบ้ข้างในไปตอบเป็นข้อเท็จจริง เช่น

        context : รหัสผ่าน Wi-Fi: <<รอเติม: รับรหัสได้ที่ไหน หรือแจ้งตอนเช็คอิน>>
        คำตอบ   : "รับรหัสได้ที่แผนกต้อนรับตอนเช็คอิน"   <-- แต่งขึ้นจากคำใบ้
    """
    path = tmp_path / "facilities.md"
    path.write_text(
        "# สิ่งอำนวยความสะดวก\n\n"
        "มี Wi-Fi ให้ใช้ฟรีทุกห้องพัก ไม่มีค่าใช้จ่ายเพิ่มเติมแต่อย่างใด\n"
        "รหัสผ่าน Wi-Fi: <<รอเติม: รับรหัส Wi-Fi ได้ที่ไหน หรือแจ้งตอนเช็คอิน>>\n"
        "รถรับส่ง: <<รอเติม: มีบริการรถรับส่งหรือไม่>>\n",
        encoding="utf-8",
    )
    document = load_document(path)
    combined = " ".join(c.text for c in chunk_document(document))

    assert "<<" not in combined
    assert "แจ้งตอนเช็คอิน" not in combined, "ข้อความใบ้ใน placeholder หลุดเข้า chunk"
    assert "รถรับส่ง" not in combined, "หัวข้อที่ยังไม่มีข้อมูลต้องไม่เหลือค้างให้ LLM เติมเอง"
    assert "มี Wi-Fi ให้ใช้ฟรีทุกห้องพัก" in combined, "ข้อมูลจริงต้องยังอยู่ครบ"

    # แต่ --todo ต้องยังนับ placeholder ได้ เพื่อบอกว่าเหลืออะไรต้องเติมบ้าง
    assert len(find_placeholders([document])) == 2


def test_เอกสารจริงต้องไม่มี_placeholder_หลุดเข้า_chunk() -> None:
    combined = " ".join(c.text for c in chunk_directory(REAL_DATA_DIR))
    assert "<<" not in combined and ">>" not in combined


@pytest.mark.skipif(not REAL_DATA_DIR.is_dir(), reason="ยังไม่มีโฟลเดอร์ data/")
def test_เอกสารจริงใน_data_มีครบทั้งสองภาษา() -> None:
    """กันลืมเพิ่มไฟล์ฝั่งใดฝั่งหนึ่งเวลาเติมเอกสารใหม่"""
    by_language: dict[str, set[str]] = {}
    for document in load_documents(REAL_DATA_DIR):
        by_language.setdefault(document.language, set()).add(Path(document.source).name)

    assert set(by_language) == set(config.SUPPORTED_LANGUAGES)
    thai_files, english_files = by_language["th"], by_language["en"]
    assert thai_files == english_files, (
        f"ไฟล์ไม่ตรงกันระหว่างสองภาษา — มีเฉพาะไทย: {thai_files - english_files} "
        f"| มีเฉพาะอังกฤษ: {english_files - thai_files}"
    )


# --------------------------------------------------------------------------
# หมายเหตุสำหรับผู้ดูแลข้อมูลต้องไม่ถูก index
# --------------------------------------------------------------------------


def test_blockquote_ที่เป็นโน้ตภายในต้องไม่ถูก_index(tmp_path: Path) -> None:
    """โน้ตภายในมักมีตัวอย่างคำถามลูกค้าอยู่ด้วย ถ้า index เข้าไปจะถูกค้นเจอ
    เป็นอันดับต้น ๆ แล้วแชทบอทเอาโน้ตไปตอบลูกค้าแทนคำตอบจริง
    """
    path = tmp_path / "faq.md"
    path.write_text(
        "# คำถามที่พบบ่อย\n\n"
        "> หมายเหตุสำหรับผู้ดูแลข้อมูล: ลูกค้ามักถามว่า เอาหมามาได้ป่ะ\n"
        "> ให้คงรูปแบบ ถาม/ตอบ ไว้เสมอเวลาเพิ่มคำถามใหม่\n\n"
        "ถาม: นำสัตว์เลี้ยงมาด้วยได้ไหม\n"
        "ตอบ: นำสัตว์เลี้ยงขนาดไม่เกิน 15 กิโลกรัมเข้าพักได้เฉพาะห้องพูลวิลล่าเท่านั้น\n",
        encoding="utf-8",
    )
    combined = " ".join(c.text for c in chunk_document(load_document(path)))

    assert "หมายเหตุสำหรับผู้ดูแลข้อมูล" not in combined
    assert "เอาหมามาได้ป่ะ" not in combined
    assert "นำสัตว์เลี้ยงขนาดไม่เกิน 15 กิโลกรัม" in combined


def test_เอกสารจริงไม่มีโน้ตภายในหลุดเข้า_chunk() -> None:
    combined = " ".join(c.text for c in chunk_directory(REAL_DATA_DIR))
    assert "หมายเหตุสำหรับผู้ดูแลข้อมูล" not in combined
    assert "Note for content editors" not in combined
