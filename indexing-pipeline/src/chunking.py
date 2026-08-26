"""
โมดูลแบ่งเอกสารเป็น chunk สำหรับภาษาไทย

ปัญหาเฉพาะของภาษาไทย
--------------------
ภาษาไทยไม่มีช่องว่างระหว่างคำ ("ห้องดีลักซ์ขอเตียงเสริมได้") ดังนั้น text splitter
ที่ตัดตามตัวอักษรดิบ ๆ มีโอกาสสูงที่จะตัดกลางคำ เช่น "ขอเตียงเส" + "ริมได้"
ซึ่งทำให้ทั้งสอง chunk เสียความหมายและ embedding เพี้ยน

วิธีแก้ในโมดูลนี้
-----------------
1. ใช้ ``RecursiveCharacterTextSplitter`` พร้อมลำดับตัวคั่นที่ออกแบบมาสำหรับไทย
   (ดู ``config.THAI_SEPARATORS``) โดยจุดสำคัญคือ **การเว้นวรรค (" ")**
   ในภาษาไทยเว้นวรรคคือขอบเขตของ "วลี/ประโยค" ไม่ใช่ขอบเขตของ "คำ"
   การตัดตรงเว้นวรรคจึงแทบไม่มีโอกาสตัดกลางคำเลย
2. ตัดเอกสารตามหัวข้อ markdown ก่อน แล้วค่อยแบ่งย่อยในแต่ละหัวข้อ
   ทำให้ chunk ไม่คร่อมสองหัวข้อที่ไม่เกี่ยวกัน
3. มีฟังก์ชันตรวจสอบคุณภาพ (``find_broken_boundaries``) ที่ตรวจจับการตัดกลางคำไทย
   จากสระ/วรรณยุกต์ที่ไปโผล่ต้น chunk หรือสระหน้าที่ค้างท้าย chunk
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

# --------------------------------------------------------------------------
# ตัวช่วยตรวจสอบขอบเขตคำไทย
# --------------------------------------------------------------------------

#: อักขระไทยที่ "ห้ามอยู่ต้นคำ" — สระตาม วรรณยุกต์ และเครื่องหมายกำกับ
#: ถ้า chunk ขึ้นต้นด้วยตัวใดตัวหนึ่งนี้ แปลว่าถูกตัดกลางคำแน่นอน
THAI_TRAILING_MARKS: frozenset[str] = frozenset("ะัาำิีึืฺุู็่้๊๋์ํ๎ๅๆฯ")

#: สระหน้าของไทย ต้องมีพยัญชนะตามหลังเสมอ
#: ถ้า chunk ลงท้ายด้วยตัวใดตัวหนึ่งนี้ แปลว่าพยัญชนะถัดไปถูกตัดขาดไป
THAI_LEADING_VOWELS: frozenset[str] = frozenset("เแโใไ")

#: หัวข้อแบบ markdown เช่น "# หัวข้อ" หรือ "## หัวข้อย่อย"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

#: front matter แบบ YAML อย่างง่ายที่คั่นด้วย --- ที่หัวไฟล์
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def starts_mid_word(text: str) -> bool:
    """เช็คว่าข้อความขึ้นต้นด้วยสระ/วรรณยุกต์ที่อยู่ต้นคำไม่ได้หรือไม่"""
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in THAI_TRAILING_MARKS


def ends_mid_word(text: str) -> bool:
    """เช็คว่าข้อความลงท้ายด้วยสระหน้าที่ยังขาดพยัญชนะตามหลังหรือไม่"""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in THAI_LEADING_VOWELS


def find_broken_boundaries(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """หา chunk ที่ถูกตัดกลางคำไทย

    Returns:
        list ของ (chunk_id, เหตุผล) เฉพาะ chunk ที่มีปัญหา — list ว่างคือผ่านหมด
    """
    problems: list[tuple[str, str]] = []
    for chunk in chunks:
        if starts_mid_word(chunk.text):
            problems.append((chunk.chunk_id, f"ขึ้นต้นด้วยสระ/วรรณยุกต์: {chunk.text[:30]!r}"))
        if ends_mid_word(chunk.text):
            problems.append((chunk.chunk_id, f"ลงท้ายด้วยสระหน้าค้าง: {chunk.text[-30:]!r}"))
    return problems


# --------------------------------------------------------------------------
# โครงสร้างข้อมูล
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Chunk:
    """หนึ่ง chunk ที่พร้อมนำไป embed และเก็บลง vector store"""

    chunk_id: str
    """ID ที่คำนวณแบบ deterministic — รันซ้ำกี่ครั้งก็ได้ค่าเดิม (ดู indexer.py)"""

    text: str
    """ข้อความที่จะถูก embed จริง (รวมบรรทัดบริบทหัวข้อแล้วถ้าเปิด PREPEND_HEADING_CONTEXT)"""

    source: str
    """ชื่อไฟล์ต้นทาง เช่น 'room_types.md'"""

    category: str
    """หมวดหมู่ เช่น 'ห้องพัก', 'นโยบาย', 'FAQ'"""

    heading: str = ""
    """หัวข้อ markdown ที่ chunk นี้อยู่ภายใต้"""

    chunk_index: int = 0
    """ลำดับที่ของ chunk ภายในไฟล์ (เริ่มจาก 0)"""

    def to_metadata(self) -> dict[str, str | int]:
        """แปลงเป็น metadata dict สำหรับ ChromaDB (รับเฉพาะ str/int/float/bool)"""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "category": self.category,
            "heading": self.heading,
            "chunk_index": self.chunk_index,
            "char_count": len(self.text),
        }


@dataclass(slots=True)
class SourceDocument:
    """เอกสารต้นฉบับหนึ่งไฟล์ หลังแยก front matter ออกแล้ว"""

    path: Path
    body: str
    front_matter: dict[str, str] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.path.name

    @property
    def category(self) -> str:
        """หาหมวดหมู่ตามลำดับ: front matter -> mapping ใน config -> ค่าเริ่มต้น"""
        if category := self.front_matter.get("category"):
            return category
        return config.CATEGORY_BY_FILENAME.get(self.path.stem, config.DEFAULT_CATEGORY)


# --------------------------------------------------------------------------
# อ่านไฟล์
# --------------------------------------------------------------------------


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """แยก YAML front matter อย่างง่าย (key: value บรรทัดละคู่) ออกจากเนื้อหา

    รองรับรูปแบบ::

        ---
        category: ห้องพัก
        title: ประเภทห้องพัก
        ---
        เนื้อหา...

    ถ้าไม่มี front matter จะคืน ({}, raw) ตามเดิม
    """
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip("\"'")
    return meta, raw[match.end():]


def load_document(path: Path) -> SourceDocument:
    """อ่านไฟล์เดียวเป็น SourceDocument (บังคับอ่านเป็น UTF-8 เพื่อไม่ให้ภาษาไทยเพี้ยน)"""
    raw = path.read_text(encoding="utf-8")
    front_matter, body = parse_front_matter(raw)
    return SourceDocument(path=path, body=body, front_matter=front_matter)


def load_documents(data_dir: Path | None = None) -> list[SourceDocument]:
    """อ่านไฟล์ทั้งหมดในโฟลเดอร์ data (เรียงตามชื่อไฟล์เพื่อให้ผลลัพธ์คงที่ทุกครั้ง)"""
    directory = Path(data_dir) if data_dir else config.DATA_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ข้อมูล: {directory}")

    paths = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
    return [load_document(p) for p in paths]


# --------------------------------------------------------------------------
# การแบ่ง chunk
# --------------------------------------------------------------------------


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """สร้าง text splitter ที่ตั้งค่าตัวคั่นให้เหมาะกับภาษาไทย

    Args:
        chunk_size: จำนวนตัวอักษรสูงสุดต่อ chunk (ค่าเริ่มต้นจาก config)
        chunk_overlap: จำนวนตัวอักษรที่ซ้อนทับกับ chunk ก่อนหน้า (ค่าเริ่มต้นจาก config)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=config.THAI_SEPARATORS,
        keep_separator=True,  # เก็บตัวคั่นไว้ ไม่ให้ประโยคติดกันจนอ่านไม่รู้เรื่อง
        length_function=len,  # นับเป็นตัวอักษร (ดูเหตุผลใน config.py)
        is_separator_regex=False,
    )


def split_into_sections(body: str) -> list[tuple[str, str]]:
    """แบ่งเอกสารตามหัวข้อ markdown ก่อนเป็นอันดับแรก

    ทำแบบนี้เพื่อไม่ให้ chunk เดียวคร่อมสองหัวข้อที่ไม่เกี่ยวกัน เช่น ท้ายเรื่อง
    "Superior Room" ไปติดกับต้นเรื่อง "Deluxe Room" ซึ่งจะทำให้ตอบคำถามผิดห้อง

    Returns:
        list ของ (heading, section_text) — heading เป็น "" สำหรับเนื้อหาก่อนหัวข้อแรก
        ถ้าไฟล์ไม่มีหัวข้อเลย (เช่นไฟล์ .txt ธรรมดา) จะคืน [("", body)] ทั้งก้อน
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    def flush(heading: str) -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append((heading, text))
        buffer.clear()

    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush(current_heading)
            current_heading = match.group(2).strip()
        else:
            buffer.append(line)
    flush(current_heading)

    return sections or [("", body.strip())]


def _build_context_header(category: str, heading: str) -> str:
    """สร้างบรรทัดบริบทที่จะเติมไว้หัว chunk"""
    parts = [f"หมวด: {category}"]
    if heading:
        parts.append(f"หัวข้อ: {heading}")
    return "[" + " | ".join(parts) + "]"


def chunk_document(
    document: SourceDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """แบ่งเอกสารหนึ่งไฟล์เป็น list ของ Chunk พร้อม metadata ครบ

    ขั้นตอน: แยกตามหัวข้อ -> แบ่งย่อยด้วย RecursiveCharacterTextSplitter ->
    ทิ้งชิ้นที่สั้นเกินไป -> เติมบรรทัดบริบท -> ให้ chunk_id แบบ deterministic
    """
    splitter = create_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    for heading, section_text in split_into_sections(document.body):
        for piece in splitter.split_text(section_text):
            piece = piece.strip()
            if len(piece) < config.MIN_CHUNK_CHARS:
                continue

            if config.PREPEND_HEADING_CONTEXT:
                text = f"{_build_context_header(document.category, heading)}\n{piece}"
            else:
                text = piece

            index = len(chunks)
            chunks.append(
                Chunk(
                    # deterministic: ชื่อไฟล์ + ลำดับที่ -> รันซ้ำได้ ID เดิมเสมอ
                    chunk_id=f"{document.path.stem}::{index:04d}",
                    text=text,
                    source=document.source,
                    category=document.category,
                    heading=heading,
                    chunk_index=index,
                )
            )

    return chunks


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """แบ่งหลายเอกสารรวดเดียว"""
    result: list[Chunk] = []
    for document in documents:
        result.extend(chunk_document(document, chunk_size, chunk_overlap))
    return result


def chunk_directory(
    data_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """สะดวก: อ่านทั้งโฟลเดอร์แล้วแบ่ง chunk ในคำสั่งเดียว"""
    return chunk_documents(load_documents(data_dir), chunk_size, chunk_overlap)


def strip_context_header(text: str) -> str:
    """ตัดบรรทัดบริบท "[หมวด: ... | หัวข้อ: ...]" ออก เหลือเฉพาะเนื้อหาจริง

    ใช้ตอนแสดงผลให้คนอ่าน เพราะบรรทัดบริบทมีไว้ช่วย embedding ไม่ใช่ให้คนอ่าน
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("[หมวด:") and lines[0].endswith("]"):
        return "\n".join(lines[1:]).strip()
    return text.strip()
