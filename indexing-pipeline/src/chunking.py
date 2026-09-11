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

    language: str = config.DEFAULT_LANGUAGE
    """ภาษาของ chunk ("th" / "en") ใช้กรองตอน retrieval ให้ตอบตรงภาษาที่ลูกค้าถาม"""

    def to_metadata(self) -> dict[str, str | int]:
        """แปลงเป็น metadata dict สำหรับ ChromaDB (รับเฉพาะ str/int/float/bool)"""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "category": self.category,
            "language": self.language,
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

    base_dir: Path | None = None
    """โฟลเดอร์ data ที่ใช้เป็นฐานคำนวณ ``source`` แบบ relative

    จำเป็นเมื่อเอกสารถูกแยกเป็นโฟลเดอร์ย่อย (data/th/faq.md กับ data/en/faq.md)
    เพราะถ้าใช้แค่ชื่อไฟล์ ทั้งสองจะได้ source เป็น "faq.md" เหมือนกันจน
    chunk_id ชนกันและทับกันเองใน vector store
    """

    @property
    def source(self) -> str:
        """path ของไฟล์เทียบกับโฟลเดอร์ data เช่น ``th/faq.md``"""
        if self.base_dir is not None:
            try:
                return self.path.relative_to(self.base_dir).as_posix()
            except ValueError:
                pass
        return self.path.name

    @property
    def slug(self) -> str:
        """ชื่อย่อที่ใช้ตั้ง chunk_id — ไม่ซ้ำกันข้ามโฟลเดอร์ เช่น ``th-faq``"""
        return self.source.rsplit(".", 1)[0].replace("/", "-")

    @property
    def category(self) -> str:
        """หาหมวดหมู่ตามลำดับ: front matter -> mapping ใน config -> ค่าเริ่มต้น"""
        if category := self.front_matter.get("category"):
            return category
        return config.CATEGORY_BY_FILENAME.get(self.path.stem, config.DEFAULT_CATEGORY)

    @property
    def language(self) -> str:
        """หาภาษาตามลำดับ: front matter -> ชื่อโฟลเดอร์ที่ไฟล์อยู่ -> ค่าเริ่มต้น"""
        if language := self.front_matter.get("language"):
            return language.lower()
        if self.path.parent.name.lower() in config.SUPPORTED_LANGUAGES:
            return self.path.parent.name.lower()
        return config.DEFAULT_LANGUAGE


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


def load_document(path: Path, base_dir: Path | None = None) -> SourceDocument:
    """อ่านไฟล์เดียวเป็น SourceDocument (บังคับอ่านเป็น UTF-8 เพื่อไม่ให้ภาษาไทยเพี้ยน)

    Args:
        path: ไฟล์ที่จะอ่าน
        base_dir: โฟลเดอร์ data ที่ใช้คำนวณ ``source`` แบบ relative
            ถ้าไม่ระบุจะใช้แค่ชื่อไฟล์
    """
    raw = path.read_text(encoding="utf-8")
    front_matter, body = parse_front_matter(raw)
    return SourceDocument(path=path, body=body, front_matter=front_matter, base_dir=base_dir)


def load_documents(data_dir: Path | None = None) -> list[SourceDocument]:
    """อ่านไฟล์ทั้งหมดในโฟลเดอร์ data รวมถึงโฟลเดอร์ย่อยตามภาษา (th/, en/)

    เรียงตาม path เพื่อให้ผลลัพธ์คงที่ทุกครั้ง
    """
    directory = Path(data_dir) if data_dir else config.DATA_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ข้อมูล: {directory}")

    paths = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
    return [load_document(p, base_dir=directory) for p in paths]


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


def strip_editor_notes(body: str) -> str:
    """ตัดบรรทัด blockquote (``>``) ซึ่งใช้เป็น "หมายเหตุสำหรับผู้ดูแลข้อมูล" ออก

    เอกสารใน ``data/`` มีคำอธิบายวิธีกรอกข้อมูลแทรกไว้ให้คนที่มาแก้ไฟล์อ่าน
    ข้อความพวกนี้ต้องไม่ถูก index เพราะมันไม่ใช่ความรู้ที่จะตอบลูกค้า
    และแย่กว่านั้นคือมันมักมีตัวอย่างคำถามของลูกค้าอยู่ด้วย ("เอาหมามาได้ป่ะ")
    ทำให้ถูกค้นเจอเป็นอันดับต้น ๆ แล้วแชทบอทเอาโน้ตภายในไปตอบลูกค้าแทนคำตอบจริง
    """
    return "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith(">")
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


#: คำที่ใช้ในบรรทัดบริบท แยกตามภาษาของเอกสาร
_CONTEXT_LABELS: dict[str, tuple[str, str]] = {
    "th": ("หมวด", "หัวข้อ"),
    "en": ("Category", "Section"),
}


def _build_context_header(category: str, heading: str, language: str) -> str:
    """สร้างบรรทัดบริบทที่จะเติมไว้หัว chunk

    ใช้คำในภาษาเดียวกับตัวเอกสาร เพราะถ้าเอาหัวข้อภาษาไทยไปแปะหน้าเนื้อหาภาษาอังกฤษ
    chunk นั้นจะกลายเป็นสองภาษาปนกัน ซึ่งทำให้เวกเตอร์จับใจความได้แย่ลง
    """
    category_label, heading_label = _CONTEXT_LABELS.get(language, _CONTEXT_LABELS["th"])
    parts = [f"{category_label}: {category}"]
    if heading:
        parts.append(f"{heading_label}: {heading}")
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

    body = strip_placeholder_lines(strip_editor_notes(document.body))
    for heading, section_text in split_into_sections(body):
        for piece in splitter.split_text(section_text):
            piece = piece.strip()
            if len(piece) < config.MIN_CHUNK_CHARS:
                continue

            if config.PREPEND_HEADING_CONTEXT:
                header = _build_context_header(document.category, heading, document.language)
                text = f"{header}\n{piece}"
            else:
                text = piece

            index = len(chunks)
            chunks.append(
                Chunk(
                    # deterministic: path ของไฟล์ + ลำดับที่ -> รันซ้ำได้ ID เดิมเสมอ
                    # ใช้ slug ที่มีโฟลเดอร์ภาษาติดมาด้วย (th-faq) ไม่ใช่แค่ชื่อไฟล์ (faq)
                    # มิฉะนั้น data/th/faq.md กับ data/en/faq.md จะได้ ID ชนกันแล้วทับกันเอง
                    chunk_id=f"{document.slug}::{index:04d}",
                    text=text,
                    source=document.source,
                    category=document.category,
                    heading=heading,
                    chunk_index=index,
                    language=document.language,
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


#: ขึ้นต้นของบรรทัดบริบท แยกตามภาษา (ต้องตรงกับ _CONTEXT_LABELS)
_CONTEXT_HEADER_PREFIXES: tuple[str, ...] = tuple(
    f"[{category_label}:" for category_label, _ in _CONTEXT_LABELS.values()
)


def strip_context_header(text: str) -> str:
    """ตัดบรรทัดบริบท "[หมวด: ... | หัวข้อ: ...]" ออก เหลือเฉพาะเนื้อหาจริง

    ใช้ตอนแสดงผลให้คนอ่าน เพราะบรรทัดบริบทมีไว้ช่วย embedding ไม่ใช่ให้คนอ่าน
    """
    lines = text.splitlines()
    if lines and lines[0].startswith(_CONTEXT_HEADER_PREFIXES) and lines[0].endswith("]"):
        return "\n".join(lines[1:]).strip()
    return text.strip()


# --------------------------------------------------------------------------
# ตรวจ placeholder ในเอกสารร่าง
# --------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(config.PLACEHOLDER_PATTERN)


def find_placeholders(documents: list[SourceDocument]) -> list[tuple[str, str]]:
    """หา placeholder ``<<...>>`` ที่ยังค้างอยู่ใน **เอกสารต้นฉบับ**

    ต้องอ่านจากเอกสารต้นฉบับ ไม่ใช่จาก chunk เพราะ ``chunk_document`` ตัดบรรทัดที่มี
    placeholder ทิ้งไปแล้ว (ดู ``strip_placeholder_lines``) ถ้าไปนับจาก chunk จะได้ 0 เสมอ

    Returns:
        list ของ (ชื่อไฟล์, placeholder ที่เจอ) — หนึ่งรายการต่อหนึ่ง placeholder
    """
    found: list[tuple[str, str]] = []
    for document in documents:
        for match in _PLACEHOLDER_RE.findall(document.body):
            found.append((document.source, match))
    return found


def strip_placeholder_lines(body: str) -> str:
    """ตัดบรรทัดที่ยังมี placeholder ``<<...>>`` ออกก่อนนำไป chunk

    **เหตุผล (วัดจากการทดลองจริงกับ Typhoon):**
    ตอนแรกเราปล่อยให้ placeholder ติดไปกับ chunk แล้วหวังว่า LLM จะเห็นแล้วรู้ว่า
    "ไม่มีข้อมูล" แต่ผลจริงตรงกันข้าม — LLM เอา **ข้อความใบ้ข้างใน placeholder**
    ไปตอบเป็นข้อเท็จจริง เช่น

        context : รหัสผ่าน Wi-Fi: <<รอเติม: รับรหัสได้ที่ไหน หรือแจ้งตอนเช็คอิน>>
        คำตอบ   : "รับรหัสได้ที่แผนกต้อนรับตอนเช็คอิน"    <-- แต่งขึ้นจากคำใบ้

    อาการนี้จับได้ยากกว่าการพ่น "<<รอเติม: ...>>" ออกมาตรง ๆ มาก เพราะคำตอบดูสมเหตุสมผล
    การตัดทิ้งทำให้ไม่มีบรรทัดนั้นใน context เลย LLM จึงตอบว่าไม่มีข้อมูลได้ถูกต้อง

    ตัด **ทั้งบรรทัด** ไม่ใช่แค่ช่วง ``<<...>>`` เพราะถ้าเหลือแต่หัวข้อค้างไว้
    (เช่น "รถรับส่ง:") LLM ก็ยังเติมคำตอบต่อท้ายให้อยู่ดี
    ผลข้างเคียงคือถ้าบรรทัดหนึ่งมีทั้งข้อมูลจริงและ placeholder ปนกัน ข้อมูลจริงจะหายไปด้วย
    ซึ่งยอมรับได้ เพราะข้อมูลขาดยังดีกว่าข้อมูลที่ถูกแต่งขึ้น
    """
    return "\n".join(
        line for line in body.splitlines() if not _PLACEHOLDER_RE.search(line)
    )
