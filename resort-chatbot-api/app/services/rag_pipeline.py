"""
ประกอบ flow การตอบคำถามทั้งหมดเข้าด้วยกัน

**เดิมเป็น RAG รอบเดียว ตอนนี้เป็นลูป tool calling**
ของเดิม: embed คำถาม -> ค้น vector store -> ยัด context เข้า prompt -> เรียก LLM ครั้งเดียว
ซึ่งตอบได้เฉพาะสิ่งที่ถูก index ไว้ล่วงหน้า จึงตอบ "ห้องว่างไหม" ไม่ได้เลย
เพราะคำตอบเปลี่ยนทุกครั้งที่มีคนจอง และ "ราคาเท่าไหร่" ก็ตอบจากข้อความที่เน่าได้

ตอนนี้: ให้โมเดลเลือกเองว่าจะถามที่ไหน ระหว่างระบบจองจริง (PostgreSQL ผ่าน Booking API)
กับคลังความรู้ แล้ววนจนกว่าจะได้คำตอบ ชื่อคลาสยังเป็น RAGPipeline เพื่อไม่ให้ client
และ dependency override เดิมพัง แต่เนื้อในเป็นลูป agent แล้ว

**ทำไมต้องจำกัดจำนวนรอบ** — โมเดลเรียก tool ที่คืน error แล้วลองใหม่วนไปเรื่อย ๆ ได้
ครบรอบแล้วเราบังคับให้สรุปด้วยการเรียกซ้ำโดยไม่ส่ง tools ไปด้วย ซึ่งทำให้มันต้อง
ตอบเป็นข้อความเท่านั้น ดีกว่าปล่อยให้ค่า token บานหรือคืน response เปล่า
"""

import json
import re
from datetime import date, datetime, timedelta, timezone

from app.config import settings
from app.core.prompts import LANGUAGE_STYLE, SYSTEM_PROMPT, no_context_answer
from app.core.tools import TOOL_SCHEMAS, ToolExecutor, guests_mentioned
from app.schemas.chat import ChatResponse, Source, SuggestedAction, TokenUsage
from app.services.booking_api import BookingAPIClient, get_booking_api_client
from app.services.chat_history import ChatHistoryStore, get_chat_history_store
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStore, get_vector_store

#: ขึ้นต้นของบรรทัดบริบทที่ indexing-pipeline เติมไว้หัว chunk
#: (ดู chunking._build_context_header ในโปรเจกต์ indexing-pipeline)
#: บรรทัดนี้มีไว้ช่วย embedding ไม่ใช่ให้คนอ่าน จึงต้องตัดออกก่อนส่งกลับเป็น snippet
_CONTEXT_HEADER_PREFIXES = ("[หมวด:", "[Category:")

#: เวลาประเทศไทย UTC+7 — เขียนเป็น offset ตายตัวแทนการใช้ ZoneInfo
#: เพราะไทยไม่มี DST ค่านี้จึงถูกต้องตลอดปี และไม่ต้องพึ่ง tzdata ที่ Windows ไม่มีมาให้
_THAI_TIMEZONE = timezone(timedelta(hours=7))

#: tool ที่ดึงข้อมูลสดจากระบบจอง (ไม่ใช่คลังความรู้)
_LIVE_DATA_TOOLS = frozenset(
    {
        "search_available_rooms",
        "quote_price",
        "get_payment_info",
        "list_restaurants",
        "get_my_bookings",
        "get_booking_payment",
    }
)


def _clean_snippet(text: str, limit: int = 200) -> str:
    """ตัดบรรทัดบริบทออกแล้วย่อให้สั้นพอสำหรับแสดงเป็นแหล่งอ้างอิง"""
    lines = text.splitlines()
    if lines and lines[0].startswith(_CONTEXT_HEADER_PREFIXES) and lines[0].endswith("]"):
        lines = lines[1:]
    cleaned = " ".join(" ".join(lines).split())
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


def _today_in_thailand() -> str:
    return datetime.now(_THAI_TIMEZONE).strftime("%Y-%m-%d")


_TH_WEEKDAYS = ("จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์")
_CALENDAR_DAYS = 14


def _calendar(today: str) -> str:
    """ปฏิทิน 2 สัปดาห์ข้างหน้าพร้อมชื่อวัน ให้โมเดลอ่านแทนการนับวันเอง

    ทดสอบจริง "เสาร์อาทิตย์หน้า" ได้ช่วงของสุดสัปดาห์นี้ แล้วตอบว่า F1 ว่าง
    ทั้งที่สุดสัปดาห์หน้า F1 มีคนจองแล้ว โมเดลนับวันในสัปดาห์จากวันที่ไม่แม่น
    """
    start = date.fromisoformat(today)
    yesterday = start - timedelta(days=1)
    lines = [
        f"- ปีนี้ = ค.ศ. {start.year} (พ.ศ. {start.year + 543}) "
        f"ปีหน้า = ค.ศ. {start.year + 1} (พ.ศ. {start.year + 544})",
        f"- {yesterday.isoformat()} วัน{_TH_WEEKDAYS[yesterday.weekday()]} (เมื่อวาน ผ่านไปแล้ว)",
    ]
    for offset in range(_CALENDAR_DAYS):
        day = start + timedelta(days=offset)
        tag = " (วันนี้)" if offset == 0 else " (พรุ่งนี้)" if offset == 1 else ""
        lines.append(f"- {day.isoformat()} วัน{_TH_WEEKDAYS[day.weekday()]}{tag}")
    saturday = start + timedelta(days=(5 - start.weekday()) % 7)  # วันเสาร์ถัดไป (หรือวันนี้ถ้าเป็นเสาร์)
    lines.append(
        f"- เสาร์อาทิตย์นี้ = เข้า {saturday.isoformat()} ออก {(saturday + timedelta(days=2)).isoformat()}"
    )
    lines.append(
        f"- เสาร์อาทิตย์หน้า = เข้า {(saturday + timedelta(days=7)).isoformat()} "
        f"ออก {(saturday + timedelta(days=9)).isoformat()}"
    )
    return "\n".join(lines)


_THAI_CHAR_RE = re.compile(r"[฀-๿]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _detect_language(message: str, requested: str) -> str:
    """ตอบตามภาษาที่ลูกค้าพิมพ์จริง ไม่ใช่ตามค่าที่แอปส่งมาอย่างเดียว

    ทดสอบจริง: แอปตั้งเป็นไทยแต่ลูกค้าถาม "Sorry, can you answer in English?"
    บอทตอบว่า "I can only communicate in Thai" ส่วนแอปตั้งเป็นอังกฤษแต่ลูกค้าพิมพ์ไทย
    "มีที่จอดรถไหม" ก็ได้คำตอบเป็นอังกฤษ (และค้นคลังความรู้ฝั่งอังกฤษจนหาไม่เจอ)
    ภาษาอื่นที่ใช้ตัวอักษรละติน (เช่น มาเลย์) ตอบเป็นอังกฤษ ซึ่งลูกค้าน่าจะอ่านออกมากกว่าไทย
    """
    if _THAI_CHAR_RE.search(message):
        return "th"
    if _LATIN_WORD_RE.search(_URL_RE.sub(" ", message)):
        return "en"
    return requested


#: กระทุ้งเมื่อโมเดลตอบโดยไม่เรียก tool เลย
#:
#: เดิมกระทุ้งเฉพาะตอนที่ตอบด้วยประโยคปฏิเสธ (เจอกับ "เรามา 3 วัน นอนเตียงเดี่ยว ราคาเท่าไหร่"
#: ประมาณ 1 ใน 4 รอบ) แต่รอบทดสอบ 134 เคสเจอฝั่งตรงข้ามด้วย คือโมเดลตอบเองจากความรู้ทั่วไป
#: โดยไม่ดูข้อมูลเลย เช่น "ราคารวม VAT แล้ว" "พนักงานพูดไทยกับอังกฤษ" "ชาบูเดอะฮัค ไม่มีร้านนี้"
#: และ "ผู้ใหญ่ 2 เด็ก 2" ที่ตอบกลับเป็นภาษาจีน จึงกระทุ้งทุกคำตอบที่ไม่เรียก tool
#: ยกเว้นการทักทายสั้น ๆ กระทุ้งแค่ครั้งเดียว ถ้ายังยืนยันคำตอบเดิมก็ปล่อยไป
_NO_TOOL_NUDGE = (
    "คุณตอบโดยยังไม่ได้เรียก tool เลย คำถามเรื่องห้องพัก ราคา การจอง การชำระเงิน นโยบาย "
    "บริการ สิ่งอำนวยความสะดวก สถานที่ หรือร้านอาหาร ต้องเรียก tool ที่เกี่ยวข้องก่อน "
    "แล้วตอบใหม่จากผลนั้นเท่านั้น (ไม่ระบุวันที่ให้ใช้ช่วงวันนี้ถึงพรุ่งนี้) "
    "ถ้าเป็นการทักทาย คุยเล่น หรือเรื่องที่ไม่เกี่ยวกับรีสอร์ทเลย ให้ตอบคำตอบเดิมได้"
)

#: กระทุ้งเมื่อโมเดลปฏิเสธทั้งที่เรียก tool ไปแล้ว — ต้องยกประโยคจากผล tool มาอ้างด้วย
#:
#: รอบทดสอบ 134 เคส ประมาณ 15 เคสตอบ "ไม่มีข้อมูล" ทั้งที่ค้นเจอข้อมูลแล้ว ที่เจอบ่อยสุดคือ
#: ข้อมูลเชิงปฏิเสธ ("ไม่มีสระว่ายน้ำ" "ไม่มีบริการนวด") ซึ่งโมเดลตีความว่าเป็น "ไม่มีข้อมูล"
#: เวอร์ชันแรกของข้อความนี้บอกแค่ "ถ้าข้อมูลบอกว่าไม่มี ให้ตอบว่าไม่มี" ผลคือแก้สระว่ายน้ำได้
#: แต่โมเดลเริ่มตอบ "ไม่มี" กับเรื่องที่ยังเป็น <<รอเติม>> (เงินประกัน ซักผ้า ปาร์ตี้ น้ำอุ่น)
#: จึงบังคับให้ยกประโยคต้นทางมาด้วย แล้วโค้ดตรวจว่าประโยคนั้นมีอยู่จริงและตรงเรื่อง
#: (ดู _verify_citation) ถ้าตรวจไม่ผ่านจะใช้ประโยคปฏิเสธแทน
_CITATION_LABEL = "อ้างอิง:"
_REFUSAL_RECHECK_NUDGE = (
    "ก่อนปฏิเสธ ให้ตรวจผล tool อีกครั้ง ถ้ามีประโยคที่ตอบคำถามนี้ได้ตรง ๆ "
    "(รวมถึงประโยคที่เขียนว่ารีสอร์ท 'ไม่มี' หรือ 'ไม่รับ' สิ่งที่ลูกค้าถาม) ให้ตอบจากประโยคนั้น "
    f"แล้วขึ้นบรรทัดใหม่เป็นบรรทัดสุดท้ายว่า '{_CITATION_LABEL} ' ตามด้วยประโยคนั้นคัดลอกจากผล tool ตรงตัว "
    "ประโยคที่เขียนว่า 'ยังไม่ได้รับข้อมูลจากรีสอร์ท' แปลว่ายังไม่มีข้อมูล ใช้อ้างอิงไม่ได้ "
    "ถ้าคำถามเรื่องร้านอาหารหรือของกินให้เรียก list_restaurants "
    "ถ้าเรื่องนโยบาย บริการ สิ่งอำนวยความสะดวก หรือขนาดห้อง ให้เรียก search_knowledge_base "
    "(ลองคำค้นอื่นได้) ถ้าตรวจแล้วไม่มีประโยคที่ตอบได้จริง ๆ ให้ตอบประโยคปฏิเสธเดิมโดยไม่ต้องมีอ้างอิง"
)

#: ปฏิเสธแบบเรียบเรียงเอง เช่น "ขออภัยค่ะ ไม่มีข้อมูลเรื่องขนาดห้องในระบบ"
#: (ทดสอบจริง "ห้องกว้างกี่ตารางเมตร" ตอบแบบนี้ทั้งที่คลังความรู้เขียนว่า 16 ตารางเมตร)
_SOFT_REFUSAL_RE = re.compile(r"ไม่มีข้อมูล|do(?:n't| not) have (?:that|this|any) information", re.I)
_SOFT_REFUSAL_MAX_LEN = 120

#: คำที่ตัดออกจากคำถามก่อนหาว่าประโยคอ้างอิงพูดถึงเรื่องเดียวกันไหม
_TOPIC_STOPWORDS_TH = (
    "ไหม", "มั้ย", "ได้", "ไม่", "บริการ", "ให้", "ห้องพัก", "ห้อง", "ที่", "ของ", "คะ", "ครับ",
    "ค่ะ", "หน่อย", "รีสอร์ท", "จะ", "เรา", "อยาก", "ขอ", "หรือ", "และ", "คน", "กี่", "เท่าไหร่",
    "ยังไง", "อะไร", "บ้าง", "แถวนี้", "นี้", "เมื่อกี้", "พนักงาน", "บอก", "ว่า", "ใช่", "มี",
)
_TOPIC_STOPWORDS_EN = frozenset(
    {"there", "have", "your", "with", "what", "does", "this", "that", "from", "about",
     "resort", "room", "rooms", "please", "could", "would", "bring", "much"}
)
_CITATION_RE = re.compile(rf"^\s*\**(?:{_CITATION_LABEL}|Source:)\**\s*(.+?)\s*$", re.M)

#: ประโยคที่แปลว่า "ยังไม่มีข้อมูล" ห้ามใช้เป็นหลักฐาน
#: indexing-pipeline แทน <<รอเติม: ...>> ด้วย "ยังไม่ได้รับข้อมูลจากรีสอร์ท ไม่มีข้อมูลเรื่องนี้ในระบบ"
#: ตอน index (ดู chunking.mask_placeholder_lines) คลังความรู้ที่บอทเห็นจริงจึงไม่มีคำว่ารอเติมเลย
_MISSING_DATA_MARKERS = ("รอเติม", "TODO", "ยังไม่ได้รับข้อมูล", "ไม่มีข้อมูลเรื่องนี้")

#: คำที่ลูกค้าใช้เรียกของในห้อง จับคู่กับ amenities ที่ระบบจองเก็บเป็นอังกฤษ
_AMENITY_WORDS = {
    "Mini fridge": ("ตู้เย็น", "fridge", "refrigerator"),
    "TV": ("ทีวี", "โทรทัศน์", "tv"),
    "Air conditioning": ("เครื่องปรับอากาศ", "แอร์", "air con"),
    "Coffee": ("กาแฟ", "coffee"),
    "Wi-Fi": ("wi-fi", "wifi", "ไวไฟ"),
}
_NEGATION_WORDS = ("ไม่มี", "no ", "not ", "without")


def _false_amenity_claims(answer_text: str, room_amenities: dict[str, list[str]]) -> list[str]:
    """ประโยคที่บอกว่าห้องหนึ่งมีของที่ amenities ของห้องนั้นไม่มี

    ทดสอบจริงสองรอบติด: ถาม "ห้องที่จองไว้มีตู้เย็นไหม" บอทตอบ "ห้อง F1 มีตู้เย็นค่ะ"
    ทั้งที่ผล get_my_bookings ของ F1 ไม่มีตู้เย็น (โมเดลไปปนกับ P1 ในผลห้องว่าง)
    ตรวจเฉพาะประโยคที่พูดถึงห้องเดียว เพื่อไม่ให้ประโยคที่ไล่หลายห้องจับผิดห้อง
    """
    claims: list[str] = []
    for sentence in re.split(r"[\n.!?]|ค่ะ|คะ", answer_text):
        rooms = set(_ROOM_NAME_RE.findall(sentence)) & set(room_amenities)
        if len(rooms) != 1:
            continue
        room = rooms.pop()
        lowered = sentence.lower()
        for amenity, words in _AMENITY_WORDS.items():
            for word in words:
                position = lowered.find(word)
                if position == -1:
                    continue
                negated = any(neg in lowered[max(0, position - 12):position] for neg in _NEGATION_WORDS)
                if not negated and amenity not in room_amenities[room]:
                    claims.append(f"ห้อง {room} ไม่มี{words[0]}")
                break
    return claims


def _normalize(text: str) -> str:
    return re.sub(r"\s+|\\n|\\\"|[\"“”'*]", "", text)


def _shares_topic(question: str, quote: str) -> bool:
    """ประโยคอ้างอิงพูดถึงสิ่งที่ลูกค้าถามจริงไหม

    กันเคสที่โมเดลยกประโยคปฏิเสธเรื่องอื่นมาอ้าง เช่น ถาม "ซักผ้า" แล้วยก
    "ไม่มีบริการรับฝากกระเป๋า" มาเป็นหลักฐานว่าไม่มีบริการซักผ้า
    ภาษาไทยไม่มีเว้นวรรค จึงเทียบเป็นชิ้นละ 3 ตัวอักษรของคำถามที่ตัดคำทั่วไปออกแล้ว
    """
    lowered = question.lower()
    for word in _TOPIC_STOPWORDS_TH:
        lowered = lowered.replace(word, " ")
    quote_lower = quote.lower()
    for word in re.findall(r"[a-z]{4,}", lowered):
        if word not in _TOPIC_STOPWORDS_EN and word in quote_lower:
            return True
    for chunk in re.sub(r"[^฀-๿]", " ", lowered).split():
        if any(chunk[i:i + 3] in quote for i in range(len(chunk) - 2)):
            return True
    return False


def _verify_citation(answer_text: str, question: str, tool_outputs: list[str]) -> str | None:
    """คืนคำตอบที่ตัดบรรทัดอ้างอิงออกแล้ว ถ้าอ้างอิงมีอยู่จริงในผล tool และตรงเรื่อง ไม่งั้นคืน None"""
    matches = list(_CITATION_RE.finditer(answer_text))
    if not matches:
        return None
    citation = matches[-1]
    quote = citation.group(1)
    body = (answer_text[: citation.start()] + answer_text[citation.end():]).strip()
    normalized_quote = _normalize(quote)
    if len(normalized_quote) < 4 or any(marker in quote for marker in _MISSING_DATA_MARKERS):
        return None
    if normalized_quote not in _normalize("\n".join(tool_outputs)):
        return None
    if not _shares_topic(question, quote):
        return None
    return body or None

#: ชื่อห้องแบบที่ระบบจองใช้ (F1, P3) — ใช้จับชื่อห้องที่โมเดลแต่งขึ้น
_ROOM_NAME_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]\d{1,2})(?![A-Za-z0-9])")

#: "ห้อง P1 ที่คุณจองไว้" / "ห้องที่จองไว้คือห้อง P1" — ห้องในประโยคแบบนี้ต้องเป็นห้องที่ลูกค้าจองจริง
#: ทดสอบจริง: "การจองของฉัน" -> "ห้องนั้นมีตู้เย็นไหม" บอทตอบ "ห้อง P1 ที่คุณจองไว้มีตู้เย็น"
#: ทั้งที่จอง F1 (หยิบชื่อห้องมาจากผลห้องว่างที่เรียกเพิ่ม)
_BOOKED_ROOM_RE = re.compile(
    r"ห้อง\s*([A-Z]\d{1,2})\s*(?:\(|ที่)?\s*(?:คุณ|ท่าน|ลูกค้า)?\s*(?:ได้)?จองไว้"
    r"|ห้องที่\s*(?:คุณ|ท่าน|ลูกค้า)?\s*(?:ได้)?จองไว้\s*(?:คือ|เป็น|\()?\s*(?:ห้อง)?\s*([A-Z]\d{1,2})"
)

#: ชื่อ tool และชื่อ field ที่ห้ามโผล่ให้ลูกค้าเห็น (ทดสอบจริง: "ค่าเตียงเสริมจาก search_knowledge_base",
#: "ไม่มีห้องใดที่ระบุไว้ในรายการ amenities", "paymentStatus = unpaid", "สถานะ **succeeded**")
_INTERNAL_NAMES = (
    "search_knowledge_base", "search_available_rooms", "quote_price", "get_payment_info",
    "list_restaurants", "get_my_bookings", "get_booking_payment", "typeLabel", "dateRangeLabel",
    "unavailableRoomNames", "pricePerNight", "paymentStatus", "currentBookings", "stayedCount",
    "pastBookingsCount", "recentPastBookings", "grandTotal", "payInApp", "guestsToEnterInApp",
    "payAtResortForExtraBeds", "roomAmenities", "roomTypeLabel", "holdMinutes", "promptPayId",
    "customerName", "unpaidUpcomingBookings", "extraBedRatePerNight", "fitsInOneRoom",
)
#: คำที่เป็นทั้งชื่อ field และคำอังกฤษปกติ ตรวจเฉพาะคำตอบภาษาไทย ("Amenities: Wi-Fi" ในคำตอบอังกฤษใช้ได้)
_INTERNAL_WORDS_TH_ONLY = ("amenities", "capacity", "succeeded", "awaiting_verification")
_INTERNAL_NAME_RE = re.compile(
    r"(?<![A-Za-z_])(" + "|".join(map(re.escape, _INTERNAL_NAMES)) + r")(?![A-Za-z_])"
)
_INTERNAL_WORD_TH_RE = re.compile(
    r"(?<![A-Za-z_])(" + "|".join(_INTERNAL_WORDS_TH_ONLY) + r")(?![A-Za-z_])", re.I
)

#: อัตราค่าเตียงเสริมต่อคืนที่คำตอบพูดถึง ตรวจเฉพาะประโยคที่มีคำว่าเตียงเสริม
#: (ไม่งั้นจะไปจับ "คืนละ 650 บาท" ของค่าห้อง) ทดสอบจริงตอบว่า "400 บาทต่อเตียงต่อคืน"
#: ทั้งที่ 400 คือค่าเตียงเสริมรวม 2 คืน อัตราจริงคือคืนละ 200
_BED_RATE_IN_ANSWER_RE = re.compile(
    r"คืนละ\s*([\d,]+)\s*บาท"
    r"|([\d,]+)\s*บาท\s*(?:/\s*|ต่อ\s*)(?:เตียง\s*(?:/\s*|ต่อ\s*))?คืน"
    r"|([\d,]+)\s*(?:THB|baht)\s*per\s*(?:bed\s*per\s*)?night",
    re.I,
)


def _wrong_booked_rooms(answer_text: str, booked: set[str]) -> list[str]:
    if not booked:
        return []
    named = {a or b for a, b in _BOOKED_ROOM_RE.findall(answer_text)}
    return sorted(name for name in named if name and name not in booked)


def _leaked_internal_names(answer_text: str, language: str) -> list[str]:
    found = set(_INTERNAL_NAME_RE.findall(answer_text))
    if language == "th":
        found |= {m.lower() for m in _INTERNAL_WORD_TH_RE.findall(answer_text)}
    return sorted(found)


def _wrong_extra_bed_rates(answer_text: str, rate: int | None) -> list[int]:
    if rate is None:
        return []
    wrong: list[int] = []
    lowered = answer_text.lower()
    # ดูเฉพาะข้อความถัดจากคำว่าเตียงเสริมไม่เกิน 60 ตัวอักษร (และไม่ข้ามบรรทัด)
    # เพื่อไม่ให้ไปจับ "คืนละ 650 บาท" ของค่าห้องที่อยู่ก่อนหน้าในบรรทัดเดียวกัน
    for marker in re.finditer(r"เตียงเสริม|extra bed", lowered):
        window = answer_text[marker.end():marker.end() + 60].split("\n")[0]
        match = _BED_RATE_IN_ANSWER_RE.search(window)
        if match:
            value = int(next(g for g in match.groups() if g).replace(",", ""))
            if value != rate:
                wrong.append(value)
    return wrong

#: ตัวอักษรจีน ญี่ปุ่น เกาหลี — ทดสอบจริงโมเดลตอบ "请告知入住日期" ให้ลูกค้าที่ถามเป็นไทย 2 ใน 134 เคส
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

#: ช่องว่างแบบแม่แบบที่โมเดลลอกมาจากตัวอย่างหรือชื่อ field แล้วไม่ได้แทนค่า
#: เจอจริง: "เตียงเสริมคืนละ [อัตรา] บาท" และ "อัปโหลดสลิปภายใน {{holdMinutes}} นาที"
_PLACEHOLDER_RE = re.compile(
    r"\{\{[^}]*\}\}|\{[A-Za-z_]+\}|\[(?:อัตรา|total|จำนวน|ราคา|ยอด|ชื่อ|วันที่)[^\]]*\](?!\()"
)

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s)\]>]+", re.I)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(\s*((?:https?://|www\.)[^)\s]+)\s*\)", re.I)

_ROOM_PLANNING_WORDS = (
    "เตียงเสริม", "ห้องเดียว", "กี่ห้อง", "พอไหม", "พอมั้ย", "พักได้", "จองกี่", "ทั้งหมด",
    "รวม", "extra bed", "one room", "how many rooms", "how much",
)

_SMALL_TALK_WORDS = ("สวัสดี", "ขอบคุณ", "hello", "hi", "thank", "โอเค", "ok", "บาย", "bye")
_QUESTION_MARKERS = ("ไหม", "มั้ย", "?", "กี่", "เท่าไหร่", "ยังไง", "อะไร", "ที่ไหน", "ได้บ้าง")

#: เดิมแทบทุกคำตอบเรื่องห้องว่างลงท้ายว่า "หากต้องการจองแจ้งได้เลยค่ะ" ซึ่งชวนให้ลูกค้าเข้าใจว่า
#: แจ้งในแชทแล้วบอทจองให้ บอทจองไม่ได้ จึงเปลี่ยนเป็นบอกให้จองในแอป
_BOOKING_CTA_RE = re.compile(
    r"(?:หาก|ถ้า)ต้องการจอง[^\n]{0,20}?แจ้ง(?:ได้)?เลย(?:นะ)?ค่ะ"
    r"|(?:หาก|ถ้า)ต้องการจอง[^\n]{0,10}?โปรดแจ้ง[^\n]*"
    r"|ต้องการให้ช่วยจอง[^\n?]*\??"
)
_BOOKING_CTA_TH = "หากต้องการจอง สามารถจองได้ในแอปเลยค่ะ"
_BOOKING_CTA_EN_RE = re.compile(
    r"(?:Please )?let me know if you(?:'|’)?d like to (?:proceed with (?:a|the) reservation|book)[^.!]*[.!]"
    r"|Would you like (?:me )?to (?:proceed with|make|help (?:you )?with) (?:a |the |your )?"
    r"(?:reservation|booking)[^?]*\?|Would you like to proceed with booking[^?]*\?",
    re.I,
)
_BOOKING_CTA_EN = "You can book directly in the app."

_MAX_NUDGES = 4


def _is_small_talk(message: str) -> bool:
    stripped = message.strip()
    if len(stripped) <= 4:
        return True
    lowered = stripped.lower()
    return (
        len(stripped) <= 25
        and any(word in lowered for word in _SMALL_TALK_WORDS)
        and not any(marker in lowered for marker in _QUESTION_MARKERS)
    )


def _unverified_urls(answer_text: str, tool_outputs: list[str]) -> list[str]:
    sources = "\n".join(tool_outputs)
    return [url for url in _URL_RE.findall(answer_text) if url.rstrip(".,") not in sources]


def _strip_urls(answer_text: str, urls: list[str]) -> str:
    """ลบลิงก์ที่ไม่ได้มาจากผล tool ออก — ใช้เมื่อกระทุ้งแล้วโมเดลยังใส่ลิงก์แต่งมาอีก"""
    def drop_link(match: re.Match) -> str:
        return "" if match.group(2) in urls else match.group(0)

    cleaned = _MD_LINK_RE.sub(drop_link, answer_text)
    for url in urls:
        cleaned = cleaned.replace(url, "")
    lines = [line.rstrip(" :") for line in cleaned.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _rewrite_booking_cta(answer_text: str) -> str:
    answer_text = _BOOKING_CTA_RE.sub(_BOOKING_CTA_TH, answer_text)
    return _BOOKING_CTA_EN_RE.sub(_BOOKING_CTA_EN, answer_text)

#: กระทุ้งเมื่อคำตอบพูดถึงเตียงเสริมโดยยังไม่ได้ค้นคลังความรู้
#:
#: ทดสอบจริงหลังเลิกส่ง maxGuestsWithExtraBed: ถาม "มา 3 คน พักห้องเดียวได้ไหม"
#: โมเดลเรียกแค่ search_available_rooms แล้วตอบ "เตียงเสริมคืนละ 300 บาท" ทุกรอบ
#: (ที่จริงคือ 200 ซึ่งคลังความรู้เป็นเจ้าของ) ทั้งที่ทั้ง SYSTEM_PROMPT และ note ในผล tool
#: สั่งให้ค้นก่อนแล้ว จึงดักด้วยโค้ดอีกชั้นเหมือน _NO_TOOL_REFUSAL_NUDGE
#: ไม่ใส่ตัวเลขในโค้ดโดยตั้งใจ — แค่บังคับให้ไปเอาจากแหล่งจริง
_EXTRA_BED_NUDGE = (
    "คำตอบของคุณพูดถึงเตียงเสริม แต่ยังไม่ได้เรียก search_knowledge_base เลย "
    "ตัวเลขค่าเตียงเสริมในคำตอบนั้นจึงไม่ได้มาจากแหล่งจริง ให้เรียก search_knowledge_base "
    "ค้น 'ค่าเตียงเสริม' ก่อน แล้วตอบใหม่ทั้งหมดโดยใช้อัตราจากผลการค้นเท่านั้น"
)
_EXTRA_BED_WORDS = ("เตียงเสริม", "extra bed")


def _mentions_extra_bed(answer_text: str) -> bool:
    lowered = answer_text.lower()
    return any(word in lowered for word in _EXTRA_BED_WORDS)


def _is_refusal(answer_text: str, refusal: str) -> bool:
    """คำตอบนี้คือการปฏิเสธล้วน ๆ หรือเปล่า

    เดิมเช็คแค่ ``refusal in answer_text`` ซึ่งพอคำตอบสั้นและมาจากคลังความรู้อย่างเดียว
    ก็ใช้ได้ แต่พอบอทดึงข้อมูลสดมาตอบ คำตอบยาวขึ้นและเป็นหลายส่วน โมเดลจึงเริ่ม
    ตอบคำถามได้ครบแล้ว **ต่อท้าย** ด้วยประโยคปฏิเสธ (จับได้จากการทดสอบจริง
    ด้วยคำถาม "ห้องพักมีกี่แบบ ต่างกันยังไง") ผลคือ answered พลิกเป็น false
    ทั้งที่ตอบได้ แล้ว frontend จะขึ้นปุ่มโทรหารีสอร์ททับคำตอบที่ใช้งานได้จริง

    เกณฑ์ที่ถูกต้องคือ "เหลืออะไรอยู่บ้างไหมหลังตัดประโยคปฏิเสธออก"
    ถ้าไม่เหลือแปลว่าปฏิเสธจริง ถ้ายังเหลือแปลว่ามันตอบได้ แค่แถมประโยคปฏิเสธมาด้วย
    ซึ่งกรณีหลังเราปล่อยข้อความไว้ตามเดิม เพราะบางทีมันหมายถึง "ส่วนที่เหลือไม่มีข้อมูล"
    ซึ่งเป็นข้อมูลที่ลูกค้าควรได้รู้
    """
    if refusal not in answer_text:
        return False
    remainder = re.sub(r"[\W_]+", "", answer_text.replace(refusal, " "))
    return not remainder


class RAGPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
        booking_client: BookingAPIClient | None = None,
        history_store: ChatHistoryStore | None = None,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.llm_client = llm_client or get_llm_client()
        self.booking_client = booking_client or get_booking_api_client()
        self.history_store = history_store or get_chat_history_store()

    async def answer(
        self,
        session_id: str,
        message: str,
        language: str = "th",
        resort_id: str | None = None,
        customer_token: str | None = None,
    ) -> ChatResponse:
        requested = (
            language if language in settings.supported_languages else settings.default_language
        )
        language = _detect_language(message, requested)
        refusal = no_context_answer(language)
        today = _today_in_thailand()
        history = self.history_store.load(session_id)

        # จำนวนคนอาจถูกบอกไว้ตั้งแต่ข้อความก่อน ("มา 4 คน" -> "ห้องเดียวขอเตียงเสริมได้ไหม")
        guests = guests_mentioned(message)
        if guests is None:
            for previous in reversed([m["content"] for m in history if m["role"] == "user"][-2:]):
                guests = guests_mentioned(previous)
                if guests is not None:
                    break

        executor = ToolExecutor(
            booking_client=self.booking_client,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            language=language,
            customer_token=customer_token,
            today=today,
            user_message=message,
            guests_hint=guests,
        )

        language_name, language_style = LANGUAGE_STYLE.get(language, LANGUAGE_STYLE["th"])
        messages: list[dict] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    language_name=language_name,
                    language_style=language_style,
                    refusal=refusal,
                    today=today,
                    calendar=_calendar(today),
                ),
            },
            # ประวัติมีแค่คำถามกับคำตอบสุดท้าย ไม่มีผล tool ของรอบก่อน
            # โมเดลจึงต้องเรียก tool ใหม่ถ้าจะอ้างห้องว่างหรือราคา ไม่ใช่หยิบของเก่ามาตอบ
            *history,
            {"role": "user", "content": message},
        ]

        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answer_text = ""
        nudges_used: set[str] = set()
        tool_rounds = 0
        awaiting_citation = False

        for _ in range(settings.agent_max_tool_rounds + _MAX_NUDGES + 1):
            # ครบจำนวนรอบเรียก tool แล้วไม่ส่ง tools ไปอีก เพื่อบังคับให้โมเดลสรุปเป็นข้อความ
            can_call_tools = tool_rounds < settings.agent_max_tool_rounds
            result = await self.llm_client.generate(
                messages, tools=TOOL_SCHEMAS if can_call_tools else None
            )

            if result.usage:
                for key in totals:
                    totals[key] += result.usage.get(key, 0)

            if not result.tool_calls:
                answer_text = result.text.strip()
                if awaiting_citation and not _is_refusal(answer_text, refusal):
                    # คำตอบหลังถูกกระทุ้งให้ตรวจซ้ำ ต้องมีประโยคอ้างอิงที่ตรวจได้ ไม่งั้นถือว่าปฏิเสธ
                    answer_text = (
                        _verify_citation(answer_text, message, executor.tool_outputs) or refusal
                    )
                    break
                nudge = None
                if len(nudges_used) < _MAX_NUDGES:
                    nudge = self._choose_nudge(
                        answer_text=answer_text,
                        message=message,
                        executor=executor,
                        refusal=refusal,
                        language_name=language_name,
                        guests=guests,
                        can_call_tools=can_call_tools,
                        used=nudges_used,
                    )
                if nudge is None:
                    break
                key, text = nudge
                nudges_used.add(key)
                if key == "auto_search":
                    # กระทุ้งให้เรียก tool แล้วโมเดลยังปฏิเสธโดยไม่เรียกอะไรเลย (ทดสอบจริง:
                    # "ขอเบอร์โทร" "มีเพจเฟซบุ๊กไหม" "มี LINE ไหม" "ถึงตีหนึ่งได้ไหม")
                    # จึงค้นคลังความรู้ด้วยคำถามของลูกค้าให้เองแล้วส่งผลกลับไปให้ตอบใหม่
                    call_id = "auto_search"
                    arguments = {"query": message}
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "search_knowledge_base",
                                        "arguments": json.dumps(arguments, ensure_ascii=False),
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": "search_knowledge_base",
                            "content": await executor.run("search_knowledge_base", arguments),
                        }
                    )
                    tool_rounds += 1
                    continue
                if key == "refusal":
                    awaiting_citation = True
                messages.append({"role": "assistant", "content": answer_text})
                messages.append({"role": "user", "content": text})
                continue

            tool_rounds += 1

            messages.append(
                {
                    "role": "assistant",
                    "content": result.text or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in result.tool_calls
                    ],
                }
            )
            for call in result.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": await executor.run(call.name, call.arguments),
                    }
                )

        # กระทุ้งแล้วยังใส่ลิงก์ที่แต่งเองมาอีก ก็ตัดทิ้งเลย ลิงก์ผิดพาลูกค้าไปผิดที่จริง
        fake_urls = _unverified_urls(answer_text, executor.tool_outputs)
        if fake_urls:
            answer_text = _strip_urls(answer_text, fake_urls)
        answer_text = _rewrite_booking_cta(answer_text)

        # โมเดลตอบข้อความว่างเปล่าได้ในบางกรณี ซึ่งแย่กว่าการปฏิเสธตรง ๆ
        if not answer_text:
            answer_text = refusal

        answered = not _is_refusal(answer_text, refusal)
        # บันทึกเฉพาะข้อความที่ลูกค้าพิมพ์กับคำตอบที่ลูกค้าเห็นจริง
        # ข้อความกระทุ้งและคำตอบที่ถูกกระทุ้งให้ตอบใหม่อยู่แค่ใน messages ของรอบนี้
        self.history_store.append(session_id, message, answer_text)
        return self._build_response(
            session_id=session_id,
            answer_text=answer_text,
            answered=answered,
            executor=executor,
            totals=totals,
        )

    @staticmethod
    def _choose_nudge(
        answer_text: str,
        message: str,
        executor: ToolExecutor,
        refusal: str,
        language_name: str,
        guests: int | None,
        can_call_tools: bool,
        used: set[str],
    ) -> tuple[str, str] | None:
        """เลือกข้อความกระทุ้งให้โมเดลตอบใหม่ หรือ None ถ้าคำตอบใช้ได้แล้ว

        แต่ละแบบใช้ได้ครั้งเดียวต่อคำถาม เรียงจากปัญหาที่ลูกค้าเห็นชัดที่สุดก่อน
        แบบที่ต้องให้เรียก tool เพิ่มใช้ได้เฉพาะตอนที่ยังเหลือรอบเรียก tool
        """
        def pick(key: str, text: str) -> tuple[str, str] | None:
            return None if key in used else (key, text)

        if _CJK_RE.search(answer_text) and not _CJK_RE.search(message):
            nudge = pick("language", f"คำตอบของคุณมีภาษาอื่นปนอยู่ ให้ตอบใหม่ทั้งหมดเป็น{language_name}เท่านั้น")
            if nudge:
                return nudge
        # ปฏิเสธแบบเรียบเรียงเองนับเฉพาะคำตอบสั้น ๆ คำตอบยาวที่ตอบได้บางส่วนแล้วบอกว่า
        # "ส่วนที่เหลือไม่มีข้อมูล" (เช่นเรื่องคืนเงิน) เป็นคำตอบที่ดีอยู่แล้ว ห้ามไปรบกวน
        # ตัดประโยคปฏิเสธตายตัวออกก่อน คำตอบที่ตอบได้แล้วแถมประโยคนั้นมาไม่ใช่การปฏิเสธ
        remainder = answer_text.replace(refusal, "").strip()
        refused = _is_refusal(answer_text, refusal) or (
            len(remainder) < _SOFT_REFUSAL_MAX_LEN and bool(_SOFT_REFUSAL_RE.search(remainder))
        )
        if can_call_tools and not executor.called_tools and not _is_small_talk(message):
            nudge = pick("no_tool", _NO_TOOL_NUDGE)
            if nudge:
                return nudge
            if refused:
                nudge = pick("auto_search", "")
                if nudge:
                    return nudge
        placeholder = _PLACEHOLDER_RE.search(answer_text)
        if placeholder:
            nudge = pick(
                "placeholder",
                f"คำตอบมีข้อความแม่แบบ {placeholder.group(0)} ที่ยังไม่ได้แทนค่า ให้แทนด้วยตัวเลข"
                "หรือข้อมูลจริงจากผล tool (ถ้ายังไม่มีให้เรียก tool ก่อน) แล้วตอบใหม่ทั้งหมด",
            )
            if nudge:
                return nudge
        leaked = _leaked_internal_names(answer_text, executor.language)
        if leaked:
            nudge = pick(
                "internal_names",
                f"คำตอบมีชื่อในระบบ ({', '.join(leaked)}) ที่ลูกค้าไม่ควรเห็น "
                "ให้ตอบใหม่ทั้งหมดเป็นภาษาคนธรรมดา ห้ามพูดถึงชื่อ tool หรือชื่อ field",
            )
            if nudge:
                return nudge
        fake_urls = _unverified_urls(answer_text, executor.tool_outputs)
        if fake_urls:
            nudge = pick(
                "url",
                f"ลิงก์ {', '.join(fake_urls)} ไม่ได้มาจากผล tool ห้ามสร้างลิงก์ขึ้นเอง "
                "ให้ตอบใหม่โดยไม่ใส่ลิงก์ที่ไม่มีในผล tool",
            )
            if nudge:
                return nudge
        invented_rooms = sorted(
            name
            for name in set(_ROOM_NAME_RE.findall(answer_text))
            if executor.known_room_names
            and name not in executor.known_room_names
            and name not in message.upper()
        )
        if invented_rooms:
            nudge = pick(
                "rooms",
                f"ห้อง {', '.join(invented_rooms)} ไม่มีอยู่ในผล tool ห้ามแต่งชื่อห้องขึ้นเอง "
                "ให้ตอบใหม่โดยพูดถึงเฉพาะห้องที่อยู่ในผล tool",
            )
            if nudge:
                return nudge
        wrong_booked = _wrong_booked_rooms(answer_text, executor.booked_room_names)
        if wrong_booked:
            nudge = pick(
                "booked_room",
                f"ลูกค้าจองห้อง {', '.join(sorted(executor.booked_room_names))} ไว้ "
                f"ไม่ใช่ห้อง {', '.join(wrong_booked)} ให้ตอบใหม่โดยใช้ห้องและของในห้อง (roomAmenities) "
                "จากผล get_my_bookings เท่านั้น ห้ามใช้ข้อมูลจากผลห้องว่าง",
            )
            if nudge:
                return nudge
        wrong_rates = _wrong_extra_bed_rates(answer_text, executor.known_extra_bed_rate())
        if wrong_rates:
            nudge = pick(
                "bed_rate",
                f"ค่าเตียงเสริมคือคืนละ {executor.known_extra_bed_rate()} บาทต่อเตียง "
                f"แต่คำตอบเขียนอัตราเป็น {', '.join(map(str, wrong_rates))} บาท ให้ตอบใหม่ "
                "ถ้าต้องการบอกยอดรวมหลายคืน ให้เขียนว่าเป็นยอดรวม ไม่ใช่อัตราต่อคืน",
            )
            if nudge:
                return nudge
        false_amenities = _false_amenity_claims(answer_text, executor.room_amenities)
        if false_amenities:
            nudge = pick(
                "amenities",
                f"ข้อมูลของในห้องไม่ตรงกับผล tool ({', '.join(false_amenities)}) "
                "ให้ตอบใหม่โดยอ่านของในห้องจากผล tool ของห้องนั้นตรง ๆ",
            )
            if nudge:
                return nudge
        if can_call_tools and executor.called_tools and refused:
            nudge = pick("refusal", _REFUSAL_RECHECK_NUDGE)
            if nudge:
                return nudge
        if (
            can_call_tools
            and guests
            and guests not in executor.planned_guests
            and (
                any(word in message.lower() for word in _ROOM_PLANNING_WORDS)
                or _mentions_extra_bed(answer_text)
            )
        ):
            nudge = pick(
                "guests",
                f"ลูกค้ามาทั้งหมด {guests} คน ให้เรียก quote_price พร้อม guests = {guests} "
                "(ใช้วันที่ที่ลูกค้าบอก ถ้าไม่ได้บอกให้ใช้วันนี้ถึงพรุ่งนี้) "
                "แล้วตอบใหม่จากผลนั้นเท่านั้น ห้ามคิดจำนวนห้องหรือเตียงเสริมเอง",
            )
            if nudge:
                return nudge
        if (
            can_call_tools
            and "search_knowledge_base" not in executor.called_tools
            and not executor.extra_bed_rate_checked
            and _mentions_extra_bed(answer_text)
        ):
            return pick("extra_bed", _EXTRA_BED_NUDGE)
        return None

    def _build_response(
        self,
        session_id: str,
        answer_text: str,
        answered: bool,
        executor: ToolExecutor,
        totals: dict[str, int],
    ) -> ChatResponse:
        # retrieval_score สะท้อนเฉพาะคุณภาพการค้นคลังความรู้ คำถามที่ตอบจากระบบจอง
        # ล้วน ๆ จึงได้ 0.0 ตามจริง ไม่ใช่เพราะค้นไม่เจอ
        best_distance = min(
            (hit["distance"] for hit in executor.knowledge_hits if hit["distance"] is not None),
            default=None,
        )
        retrieval_score = (
            max(0.0, min(1.0, 1.0 - best_distance)) if best_distance is not None else 0.0
        )

        # ข้อมูลจากระบบจองเป็นข้อมูลจริง ณ เวลานั้น ไม่ใช่การเดาจากความใกล้เคียงของเวกเตอร์
        # คำตอบที่อิงข้อมูลนั้นจึงไม่ควรถูกลดความมั่นใจลงตามคะแนน similarity
        used_live_data = any(name in _LIVE_DATA_TOOLS for name in executor.successful_tools)
        if not answered:
            confidence = 0.0
        elif used_live_data:
            confidence = 1.0
        else:
            confidence = retrieval_score

        # ตัด doc_id ซ้ำออก เพราะโมเดลค้นคลังความรู้ได้หลายรอบในหนึ่งคำถาม
        seen: set[str] = set()
        sources: list[Source] = []
        if answered:
            for hit in executor.knowledge_hits:
                if hit["doc_id"] in seen:
                    continue
                seen.add(hit["doc_id"])
                sources.append(
                    Source(doc_id=hit["doc_id"], snippet=_clean_snippet(hit["text"]))
                )

        return ChatResponse(
            session_id=session_id,
            answer=answer_text,
            sources=sources,
            answered=answered,
            confidence=round(confidence, 2),
            retrieval_score=round(retrieval_score, 4),
            suggested_action=SuggestedAction(type="none", url=""),
            tools_used=list(executor.called_tools),
            token_usage=TokenUsage(**totals) if totals["total_tokens"] else None,
        )


def get_rag_pipeline() -> RAGPipeline:
    """Factory function สำหรับใช้กับ FastAPI dependency injection"""
    return RAGPipeline()
