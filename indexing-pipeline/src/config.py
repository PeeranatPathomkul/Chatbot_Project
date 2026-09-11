"""
ค่าคงที่และการตั้งค่าทั้งหมดของ indexing pipeline

หลักการ: logic ทุกที่ต้อง import ค่าจากไฟล์นี้ ห้าม hardcode path / ชื่อโมเดล / chunk size
ที่อื่น ทุกค่าสามารถ override ผ่าน environment variable ได้ (ดูชื่อ env ในวงเล็บ)
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

#: root ของโปรเจกต์ (โฟลเดอร์ที่มี src/, data/, chroma_db/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: โฟลเดอร์เก็บเอกสารต้นฉบับ (RESORT_DATA_DIR)
DATA_DIR: Path = Path(os.getenv("RESORT_DATA_DIR", PROJECT_ROOT / "data"))

#: โฟลเดอร์เก็บ persistent vector store ของ ChromaDB (RESORT_CHROMA_DIR)
CHROMA_DIR: Path = Path(os.getenv("RESORT_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))

#: นามสกุลไฟล์ที่ indexer จะอ่าน
SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md")


# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------

#: โมเดล embedding แบบ multilingual ที่รองรับภาษาไทย (768 มิติ) (RESORT_EMBEDDING_MODEL)
EMBEDDING_MODEL_NAME: str = os.getenv("RESORT_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

#: จำนวนมิติของเวกเตอร์ที่โมเดลด้านบนผลิตออกมา (ใช้ตรวจสอบความถูกต้องเท่านั้น)
EMBEDDING_DIMENSION: int = 768

#: ความยาวสูงสุด (token) ที่โมเดล e5-base รับได้ ข้อความที่ยาวกว่านี้จะถูกตัดทิ้งเงียบ ๆ
#: จึงต้องคุม CHUNK_SIZE ให้ประมาณการแล้วไม่เกินค่านี้
MODEL_MAX_TOKENS: int = 512

#: prefix ที่โมเดลตระกูล e5 บังคับให้ใส่ (ดู model card ของ intfloat/multilingual-e5-base)
#: ถ้าไม่ใส่หรือใส่สลับกัน คุณภาพการค้นคืนจะตกลงอย่างชัดเจน
PASSAGE_PREFIX: str = "passage: "
QUERY_PREFIX: str = "query: "

#: จำนวนข้อความต่อ batch ตอนเรียก model.encode() (RESORT_EMBED_BATCH_SIZE)
EMBED_BATCH_SIZE: int = int(os.getenv("RESORT_EMBED_BATCH_SIZE", "32"))

#: normalize เวกเตอร์ให้มีความยาว 1 เสมอ ทำให้ dot product เท่ากับ cosine similarity พอดี
NORMALIZE_EMBEDDINGS: bool = True

#: อุปกรณ์ที่ใช้รันโมเดล ("cpu", "cuda", หรือ None = ให้ sentence-transformers เลือกเอง)
EMBEDDING_DEVICE: str | None = os.getenv("RESORT_EMBEDDING_DEVICE") or None


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# เหตุผลของค่าที่เลือก อธิบายไว้ใน README หัวข้อ "ทำไมถึงเลือกค่านี้"
# สรุปสั้น ๆ: นับเป็น "จำนวนตัวอักษร" ไม่ใช่ token เพราะภาษาไทยไม่มีช่องว่างระหว่างคำ
# อัตราส่วนโดยประมาณของ tokenizer XLM-RoBERTa (ที่ e5 ใช้) คือ 1 token ~ 2-3 ตัวอักษรไทย
# ดังนั้น 450 ตัวอักษร ~ 150-220 token ปลอดภัยกว่าเพดาน 512 token มาก

#: ความยาวสูงสุดของแต่ละ chunk นับเป็นจำนวนตัวอักษร (RESORT_CHUNK_SIZE)
CHUNK_SIZE: int = int(os.getenv("RESORT_CHUNK_SIZE", "450"))

#: จำนวนตัวอักษรที่ chunk ติดกันซ้อนทับกัน ~20% ของ CHUNK_SIZE (RESORT_CHUNK_OVERLAP)
CHUNK_OVERLAP: int = int(os.getenv("RESORT_CHUNK_OVERLAP", "90"))

#: chunk ที่สั้นกว่านี้ (หลัง strip) จะถูกทิ้ง เพราะมักเป็นเศษหัวข้อหรือบรรทัดว่าง
MIN_CHUNK_CHARS: int = int(os.getenv("RESORT_MIN_CHUNK_CHARS", "40"))

#: ลำดับตัวคั่นสำหรับ RecursiveCharacterTextSplitter
#: LangChain จะพยายามตัดด้วยตัวคั่นตัวแรกก่อน ถ้า chunk ยังยาวเกินจึงไล่ลงไปตัวถัดไป
#: เรียงจาก "ขอบเขตความหมายใหญ่" ไป "เล็ก" เพื่อให้ตัดตรงรอยต่อที่เสียความหมายน้อยที่สุด
THAI_SEPARATORS: list[str] = [
    "\n\n",   # 1. ระหว่างย่อหน้า / ระหว่างหัวข้อ — ขอบเขตความหมายที่ชัดที่สุด
    "\n",     # 2. ระหว่างบรรทัด — เอกสาร FAQ และ bullet list ของไทยแยกความหมายด้วยบรรทัด
    "। ",     # 3. (สำรอง) danda สำหรับเอกสารที่ paste มาจากภาษาอื่น
    "? ",     # 4. ท้ายประโยคคำถาม
    "! ",     # 5. ท้ายประโยคอุทาน
    ". ",     # 6. จุดจบประโยคแบบตะวันตก (ไทยใช้ปนกับตัวเลข/ตัวย่อ จึงต้องมี space ตามหลัง)
    "ฯ ",     # 7. ไปยาลใหญ่ มักตามด้วยการเว้นวรรค
    " ",      # 8. เว้นวรรคไทย — สำคัญที่สุดสำหรับภาษาไทย เพราะไทย "ไม่" เว้นวรรคระหว่างคำ
              #    แต่เว้นวรรคระหว่าง "วลี/ประโยค" ตัดตรงนี้จึงแทบไม่มีโอกาสตัดกลางคำ
    "",       # 9. fallback สุดท้าย: ตัดตามตัวอักษร (เสี่ยงตัดกลางคำ ใช้เมื่อไม่มีทางเลือกอื่น)
]

#: หมวดหมู่เริ่มต้น เมื่อไฟล์ไม่ได้ระบุ category ใน front matter และไม่มีใน CATEGORY_BY_FILENAME
DEFAULT_CATEGORY: str = "ทั่วไป"

#: mapping สำรอง: ชื่อไฟล์ (ไม่รวมนามสกุล) -> หมวดหมู่
#: ใช้เมื่อไฟล์ .txt ที่ไม่มี front matter ให้ใส่ชื่อไฟล์ตรงนี้ได้เลย
CATEGORY_BY_FILENAME: dict[str, str] = {
    "overview": "ข้อมูลทั่วไป",
    "room_types": "ห้องพัก",
    "rates": "ราคา",
    "rates_and_packages": "ราคา",
    "booking_policy": "นโยบาย",
    "facilities": "สิ่งอำนวยความสะดวก",
    "location_and_travel": "การเดินทาง",
    "faq": "FAQ",
}


# --------------------------------------------------------------------------
# ภาษา
# --------------------------------------------------------------------------
# เอกสารถูกแยกเป็นโฟลเดอร์ย่อยตามภาษา (data/th/, data/en/) แทนที่จะปนกันในไฟล์เดียว
# เหตุผล: chunk หนึ่งชิ้นควรเป็นภาษาเดียวล้วน ๆ ถ้าปนสองภาษาในชิ้นเดียว เวกเตอร์จะถูก
# เฉลี่ยระหว่างสองภาษาจนจับใจความได้แย่ลงทั้งคู่ และการแยกยังทำให้กรองตอน retrieval ได้ด้วย

#: ภาษาที่รองรับ — ใช้เป็นชื่อโฟลเดอร์ย่อยใต้ data/ ด้วย
SUPPORTED_LANGUAGES: tuple[str, ...] = ("th", "en")

#: ภาษาเริ่มต้น เมื่อไฟล์ไม่ได้อยู่ในโฟลเดอร์ภาษาและไม่ระบุใน front matter
DEFAULT_LANGUAGE: str = os.getenv("RESORT_DEFAULT_LANGUAGE", "th")


# --------------------------------------------------------------------------
# Placeholder ในเอกสารร่าง
# --------------------------------------------------------------------------
# เอกสารใน data/ เป็น "โครง" ที่ยังรอเติมข้อมูลจริง ทุกจุดที่ยังไม่มีข้อมูลถูกทำเครื่องหมาย
# ไว้ด้วย <<...>> เพื่อให้ค้นหาและนับความคืบหน้าได้ง่าย
#
# สำคัญ: ถ้า index เอกสารที่ยังมี placeholder เข้าไป แชทบอทจะตอบลูกค้าด้วยข้อความ
# "<<รอเติม: ราคา>>" อย่างมั่นใจ ซึ่งแย่กว่าการตอบว่าไม่รู้มาก indexer จึงเตือนเสมอ
# เมื่อเจอ placeholder และมีธง --strict ไว้ให้หยุดทำงานไปเลยสำหรับใช้ใน CI

PLACEHOLDER_PATTERN: str = r"<<[^>]{0,200}>>"


# --------------------------------------------------------------------------
# ChromaDB
# --------------------------------------------------------------------------

#: ชื่อ collection ใน ChromaDB (RESORT_COLLECTION_NAME)
COLLECTION_NAME: str = os.getenv("RESORT_COLLECTION_NAME", "resort_knowledge")

#: metadata ของ collection — บังคับให้ HNSW ใช้ระยะทางแบบ cosine
#: สำคัญมาก: ค่าเริ่มต้นของ Chroma คือ "l2" ถ้าไม่ระบุ ขั้นตอน retrieval จะได้อันดับผิด
COLLECTION_METADATA: dict[str, str] = {"hnsw:space": "cosine"}

#: จำนวนเอกสารที่ upsert ต่อครั้ง (กัน payload ใหญ่เกินไปตอนไฟล์เยอะ)
UPSERT_BATCH_SIZE: int = int(os.getenv("RESORT_UPSERT_BATCH_SIZE", "128"))


# --------------------------------------------------------------------------
# Retrieval (ใช้เฉพาะในสคริปต์ทดสอบ ยังไม่ใช่ API จริง)
# --------------------------------------------------------------------------

#: จำนวนผลลัพธ์ที่ดึงมาตอนทดสอบ query
DEFAULT_TOP_K: int = int(os.getenv("RESORT_TOP_K", "3"))


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("RESORT_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"


# --------------------------------------------------------------------------
# Contextual chunk header
# --------------------------------------------------------------------------

#: ถ้าเป็น True จะเติมบรรทัดบริบท "[หมวด: ... | หัวข้อ: ...]" ไว้หัว chunk ก่อนนำไป embed
#: ช่วยให้ chunk ที่ถูกตัดออกมากลางเอกสารยังรู้ว่าตัวเองพูดถึงเรื่องอะไร
#: (เช่น chunk ที่เขียนว่า "ขอเตียงเสริมได้ 1 เตียง" จะรู้ว่าอยู่ใต้หัวข้อ "Deluxe Room")
PREPEND_HEADING_CONTEXT: bool = os.getenv("RESORT_PREPEND_HEADING", "1") != "0"


# --------------------------------------------------------------------------
# เกณฑ์คะแนนตอน retrieval
# --------------------------------------------------------------------------

#: คะแนน similarity ขั้นต่ำที่ถือว่า chunk "น่าจะเกี่ยวข้อง" พอจะส่งให้ LLM
#:
#: ระวัง: e5 ให้คะแนนคู่ข้อความไทยสูง 0.8+ แทบทุกคู่แม้จะไม่เกี่ยวกันเลย
#: ค่านี้จึงกรองได้แค่ของที่ไม่เกี่ยวจริง ๆ เท่านั้น ไม่ใช่เครื่องมือกันตอบมั่ว
#: การกันตอบมั่วต้องทำที่ prompt ตอน generation ด้วย (สั่งให้ตอบเฉพาะจาก context
#: และให้บอกว่าไม่ทราบเมื่อ context ไม่ครอบคลุมคำถาม)
#: ดูตัวเลขจริงได้จาก: python -m scripts.evaluate
DEFAULT_SIMILARITY_THRESHOLD: float = float(os.getenv("RESORT_SIMILARITY_THRESHOLD", "0.80"))
