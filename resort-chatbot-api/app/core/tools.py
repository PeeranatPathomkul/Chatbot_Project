"""
นิยาม tool ที่โมเดลเรียกได้ พร้อมตัวกลางที่รันคำขอนั้นจริง (ToolExecutor)

**ทำไมเอา vector search มาเป็น tool ตัวหนึ่ง แทนที่จะค้นก่อนเสมอ**
คำถามจริงของลูกค้ามักต้องใช้สองแหล่งต่อกัน เช่น "ห้องเงียบ ๆ เสาร์นี้ว่างไหม ราคาเท่าไหร่"
ต้องถามคลังความรู้ว่าห้องไหนเงียบ แล้วถามระบบจองว่าห้องนั้นว่างจริงไหม
การค้น vector ก่อนเสมอรอบเดียวตอบเคสแบบนี้ไม่ได้ ส่วนการเขียน intent classifier
มาแยกทางก็ต้องไล่เดาคู่ผสมเองไม่จบ ปล่อยให้โมเดลเลือกลำดับเองในลูป tool calling
จัดการได้ตรงกว่า และเพิ่ม tool ใหม่ทีหลังได้โดยไม่ต้องแก้ตัวแยกทาง

**เส้นแบ่งว่าข้อมูลไหนมาจากไหน**
- ระบบจอง (PostgreSQL ผ่าน Booking API) เป็นเจ้าของ:
  ห้องว่าง ราคาห้อง ชื่อห้อง จำนวนห้อง capacity สิ่งอำนวยความสะดวกในห้อง (แต่ละห้องไม่เหมือนกัน)
  เลขพร้อมเพย์ และเวลากันห้องก่อนยกเลิกอัตโนมัติ (holdMinutes)
- คลังความรู้ (Chroma) เป็นเจ้าของ:
  ค่าเตียงเสริมและวิธีขอ วิธีชำระเงินโดยรวม นโยบายยกเลิก เวลาเช็คอิน
  สิ่งอำนวยความสะดวกส่วนกลาง การเดินทาง
ตัวเลขที่ระบบจองเป็นเจ้าของถูกถอดออกจากคลังความรู้แล้ว เพื่อไม่ให้มีสองแหล่งที่ขัดกันเอง
(ถ้าเติมราคากลับเข้าไปใน data/*.md เมื่อไหร่ บอทจะเริ่มตอบราคาที่ไม่ตรงกับที่จองจริงได้)

**customer_token ต้องไม่ใช่ parameter ที่โมเดลกรอก**
มันถูกฉีดเข้ามาตอนสร้าง ToolExecutor จาก header ของ request ถ้าปล่อยให้เป็น
parameter ใน schema โมเดลจะแต่ง token ขึ้นมาเองได้ และคำสั่งที่แฝงมาในข้อความลูกค้า
ก็สั่งให้มันใช้ token ของคนอื่นได้ด้วย
"""

import json
import re
from datetime import date, timedelta
from typing import Any

from app.config import settings
from app.services.booking_api import (
    BookingAPIAuthError,
    BookingAPIClient,
    BookingAPIError,
    format_stay_dates,
    group_bookings,
    plan_rooms_for_guests,
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

#: ผลที่ส่งกลับให้โมเดลเมื่อ tool ต้องใช้สิทธิ์แต่ลูกค้ายังไม่ได้ล็อกอิน
#:
#: hint ส่วนหลังเพิ่มเพราะทดสอบจริง "ตีสองเข้าห้องไม่ได้ กุญแจหาย" กับ "ห้องสกปรก อยากร้องเรียน"
#: โมเดลเรียก get_my_bookings แล้วตอบแค่ "กรุณาเข้าสู่ระบบก่อน" ทั้งที่ลูกค้าต้องการเบอร์โทร
_LOGIN_REQUIRED = {
    "error": "not_authenticated",
    "hint": (
        "ลูกค้ายังไม่ได้ล็อกอิน ถ้าลูกค้าอยากดูข้อมูลการจองของตัวเอง ให้แจ้งว่าต้องเข้าสู่ระบบก่อน "
        "แต่ถ้าลูกค้าแจ้งปัญหา ร้องเรียน ลืมของ หรือเรื่องด่วน ไม่ต้องให้ล็อกอิน "
        "ให้ขอโทษแล้วแนะนำให้โทรหารีสอร์ทที่ 081-598-1199 ทันที"
    ),
}

#: ปีที่ห่างจากปีนี้เกินเท่านี้แทบแน่ใจว่าเป็นปี พ.ศ. ที่หลุดมาเป็น ค.ศ. (2569 -> ค.ศ. 2569)
#: เดิมใช้เพดาน 365 วัน แต่ลูกค้าที่ถาม "ตุลาคมปีหน้า" จริง ๆ โดนบอกว่าพิมพ์ผิด
#: และ backend ก็ไม่ได้จำกัดว่าจองล่วงหน้าได้ไกลแค่ไหน
_BUDDHIST_YEAR_GAP = 100
_BUDDHIST_ERA_OFFSET = 543

_DATE_ERRORS = frozenset(
    {"date_in_past", "date_looks_buddhist", "year_mismatch", "date_format_mismatch"}
)

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
_YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


_NEXT_YEAR_RE = re.compile(r"ปีหน้า|next year", re.I)
_THIS_YEAR_RE = re.compile(r"ปีนี้|this year", re.I)
_YESTERDAY_RE = re.compile(r"เมื่อวาน|yesterday", re.I)
#: "10/11" แบบไทยคือ วัน/เดือน — ทดสอบจริงโมเดลอ่านเป็น 10-11 ตุลาคมสองรอบติด
_SLASH_DATE_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/\d{2,4})?(?![\d/])")
#: "2 ห้อง 3 คืน" — ทดสอบจริงโมเดลไม่ส่ง rooms มา เลยคิดเป็นห้องเดียว
_ROOMS_RE = re.compile(r"(\d+)\s*ห้อง|(\d+)\s*rooms?", re.I)


def years_mentioned(message: str, this_year: int | None = None) -> set[int]:
    """ปี ค.ศ. ที่ลูกค้าระบุมาเองในข้อความ (แปลง พ.ศ. ให้แล้ว)

    ตัด UUID ออกก่อน เพราะ booking id มีกลุ่มตัวเลข 4 หลักที่หน้าตาเหมือนปีได้
    นับ "ปีหน้า" / "ปีนี้" ด้วย เพราะทดสอบจริง "ตุลาคมปีหน้า วันที่ 20" ได้ปีนี้ 1 ใน 2 รอบ
    """
    years: set[int] = set()
    for raw in _YEAR_RE.findall(_UUID_RE.sub(" ", message)):
        year = int(raw)
        if 2400 <= year <= 2700:
            years.add(year - _BUDDHIST_ERA_OFFSET)
        elif 1900 <= year <= 2200:
            years.add(year)
    if this_year is not None:
        if _NEXT_YEAR_RE.search(message):
            years.add(this_year + 1)
        if _THIS_YEAR_RE.search(message):
            years.add(this_year)
    return years


def _slash_dates(message: str) -> list[tuple[int, int]]:
    """(วัน, เดือน) จากตัวเลขแบบ "10/11" ในข้อความ ตัดคู่ที่เป็นวันที่ไม่ได้ทิ้ง"""
    return [
        (int(day), int(month))
        for day, month in _SLASH_DATE_RE.findall(message)
        if 1 <= int(day) <= 31 and 1 <= int(month) <= 12
    ]


_THAI_NUMBERS = {
    "หนึ่ง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5,
    "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9, "สิบ": 10,
}
_NUMBER = r"(\d+|" + "|".join(_THAI_NUMBERS) + r")"
_GUESTS_RE = re.compile(_NUMBER + r"\s*(?:คน|ท่าน|people|persons?|guests?|pax|adults?)", re.I)
_PARTY_RE = re.compile(r"(?:ผู้ใหญ่|เด็ก)\s*" + _NUMBER)


def _to_int(token: str) -> int:
    return _THAI_NUMBERS.get(token) or int(token)


def guests_mentioned(text: str) -> int | None:
    """จำนวนผู้เข้าพักที่ลูกค้าพิมพ์มาเอง ("มา 4 คน" "ผู้ใหญ่ 2 เด็ก 2" "สามคน") หรือ None"""
    party = [_to_int(n) for n in _PARTY_RE.findall(text)]
    if party:
        return sum(party)
    counts = [_to_int(n) for n in _GUESTS_RE.findall(text)]
    return max(counts) if counts else None


_NIGHTS_RE = re.compile(_NUMBER + r"\s*(?:คืน|nights?)", re.I)
_WEEKS_RE = re.compile(_NUMBER + r"\s*(?:อาทิตย์|สัปดาห์|weeks?)", re.I)


def nights_mentioned(message: str) -> int | None:
    """จำนวนคืนที่ลูกค้าบอกตรง ๆ ("3 คืน" "2 อาทิตย์") หรือ None

    ไม่นับ "3 วัน" เพราะอาจหมายถึง 2 หรือ 3 คืนก็ได้ (กติกาข้อ 7 ให้โมเดลเลือกแล้วบอกลูกค้า)
    """
    nights = [_to_int(n) for n in _NIGHTS_RE.findall(message)]
    weeks = [_to_int(n) * 7 for n in _WEEKS_RE.findall(message)]
    found = nights + weeks
    return max(found) if found else None


def rooms_mentioned(message: str) -> int | None:
    counts = [int(a or b) for a, b in _ROOMS_RE.findall(message)]
    return max(counts) if counts else None


def _check_stay_dates(check_in: str, check_out: str, today: str) -> dict[str, Any] | None:
    """คืน error ให้โมเดลถ้าช่วงวันที่ใช้ไม่ได้ คืน None ถ้าผ่าน

    **ทำไมต้องเช็คที่นี่** Booking API กันแค่รูปแบบผิดกับวันออกก่อนวันเข้า แต่วันในอดีต
    มันตอบห้องว่างกลับมาตามปกติ เจอจริง: ลูกค้าถาม "29 ตุลา 2004" บอทตอบว่า
    "วันที่ 29 ตุลาคม 2569 มีห้องว่าง" คือเปลี่ยนปีให้เองเงียบ ๆ โดยไม่ถามลูกค้า
    ถ้าคืน error พร้อม hint โมเดลจะถามกลับว่าหมายถึงปีไหน แทนที่จะเดาเอง
    """
    try:
        start = date.fromisoformat(check_in)
        end = date.fromisoformat(check_out)
    except ValueError:
        return {
            "error": "bad_arguments",
            "detail": "วันที่ต้องอยู่ในรูปแบบ YYYY-MM-DD และต้องเป็นวันที่ที่มีอยู่จริง",
            "hint": "ถ้าลูกค้าพิมพ์วันที่ที่ไม่มีจริง เช่น 31 กุมภาพันธ์ ให้บอกลูกค้าตรง ๆ แล้วถามวันที่ใหม่",
        }
    today_date = date.fromisoformat(today)
    if end <= start:
        return {"error": "bad_arguments", "detail": "วันออกต้องอยู่หลังวันเข้าพักอย่างน้อย 1 วัน"}
    if start < today_date:
        return {
            "error": "date_in_past",
            "detail": f"วันเข้าพัก {check_in} ผ่านมาแล้ว (วันนี้คือ {today}) จองย้อนหลังไม่ได้",
            "hint": (
                "ห้ามเปลี่ยนวันหรือปีเองแล้วค้นต่อ และห้ามเปลี่ยนไปค้นวันนี้แทน "
                "ให้บอกลูกค้าว่าวันที่นั้นผ่านมาแล้ว แล้วถามว่าต้องการวันไหน"
            ),
        }
    if start.year - today_date.year > _BUDDHIST_YEAR_GAP:
        return {
            "error": "date_looks_buddhist",
            "detail": f"ปี {start.year} น่าจะเป็นปี พ.ศ. ที่ยังไม่ได้แปลง",
            "hint": f"ปี พ.ศ. ต้องลบ {_BUDDHIST_ERA_OFFSET} ให้เป็น ค.ศ. ก่อนส่ง แล้วเรียก tool ใหม่",
        }
    return None


#: แนบไปกับผลของ search_available_rooms ทุกครั้ง (เหตุผลอยู่ที่ ToolExecutor._dispatch)
#:
#: ส่วนเรื่อง amenities กับ unavailableRoomNames เพิ่มหลังทดสอบจริง:
#: - "ในห้องมีไมโครเวฟไหม" -> "P1 มีไมโครเวฟ" (P1 มีตู้เย็น ไม่มีไมโครเวฟ)
#: - "มีเครื่องทำน้ำอุ่นไหม" -> แต่งว่าบางห้องมีบางห้องไม่มี ทั้งที่ไม่มีข้อมูลเลย
#: - ห้องที่มีคนจองแล้วถูกตอบว่า "ห้อง P1 ไม่มีข้อมูลในระบบ" ลูกค้าจะเข้าใจว่าไม่มีห้องนี้
_ROOMS_NOTE = (
    "capacity คือจำนวนผู้เข้าพักสูงสุดที่จองในแอปได้ต่อห้อง ถ้าลูกค้าบอกจำนวนคนมา "
    "ให้เรียก quote_price พร้อม guests เพื่อให้ระบบคิดจำนวนห้องและเตียงเสริมให้ "
    "ผลนี้ไม่มีค่าเตียงเสริม ห้ามเดาตัวเลข และห้ามแสดงชื่อ field ให้ลูกค้าเห็น "
    "ของในห้องมีเท่าที่อยู่ใน amenities ของห้องนั้นเท่านั้น ถ้าลูกค้าถามถึงของที่ไม่อยู่ใน "
    "amenities ของห้องไหนเลย ให้ตอบว่าไม่มีข้อมูลเรื่องนั้น ห้ามบอกว่าห้องไหนมี "
    "ห้องใน unavailableRoomNames คือห้องที่มีคนจองแล้วในช่วงนี้ ให้ตอบว่าห้องนั้นไม่ว่าง "
    "ห้องที่ลูกค้าถามแต่ไม่อยู่ทั้งใน rooms และ unavailableRoomNames ให้ตอบว่าห้องนั้นจองไม่ได้ในช่วงนี้ "
    "ห้ามพูดว่าไม่มีห้องนี้หรือไม่มีข้อมูลในระบบ ตอนบอกวันที่ให้ใช้ dateRangeLabel"
)

#: แปลง amenities ที่ระบบจองเก็บเป็นอังกฤษ ให้เป็นคำที่ลูกค้าพิมพ์จริง
#: (หลักเดียวกับ booking_api.ROOM_TYPE_LABELS) ตัวที่ไม่มีในนี้ส่งค่าดิบไป
_AMENITY_LABELS_TH = {
    "Wi-Fi": "Wi-Fi",
    "Air conditioning": "เครื่องปรับอากาศ",
    "TV": "ทีวี",
    "Mini fridge": "ตู้เย็นขนาดเล็ก",
    "Coffee": "ชุดชงกาแฟ",
    "Two beds": "เตียง 2 เตียง",
}

#: ดึงค่าเตียงเสริมจากข้อความในคลังความรู้ คลังความรู้ยังเป็นเจ้าของตัวเลขนี้
#: โค้ดแค่อ่านออกมาคิดให้ ถ้าวันหนึ่งเจ้าของแก้ราคาใน booking_policy.md ก็ตามไปเอง
_EXTRA_BED_RATE_RE = re.compile(r"เตียงเสริม[^\n]*?คืนละ\s*([\d,]+)\s*บาท")
_EXTRA_BED_QUERY = "ค่าเตียงเสริม คืนละกี่บาทต่อเตียง"
_MAX_EXTRA_BEDS_PER_ROOM = 1  # "ห้องละไม่เกิน 1 เตียง" ตาม booking_policy.md

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_available_rooms",
            "description": (
                "ค้นหาห้องพักที่ว่างจริงในช่วงวันที่ที่ระบุ พร้อมราคาต่อคืนและจำนวนผู้เข้าพักที่รับได้ "
                "แต่ละห้องคืนค่าที่ต้องใช้: capacity คือจำนวนผู้เข้าพักสูงสุดที่จองในแอปได้ต่อห้อง, "
                "typeLabel คือชนิดเตียงเป็นภาษาไทย ใช้กรองเมื่อลูกค้าระบุว่าอยากได้เตียงแบบไหน, "
                "และ amenities คือสิ่งอำนวยความสะดวกของห้องนั้น (แต่ละห้องไม่เหมือนกัน) "
                "ใช้ tool นี้ทุกครั้งที่ลูกค้าถามเรื่องห้องว่าง ราคาห้อง ของในห้อง หรือขอให้คิดราคารวม "
                "ต้องระบุวันที่เสมอ ถ้าลูกค้าไม่ได้บอกวันที่ ให้ใช้วันนี้ถึงพรุ่งนี้ "
                "แล้วบอกลูกค้าด้วยว่าคิดให้จากช่วงวันไหน"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "check_in": {
                        "type": "string",
                        "description": "วันเข้าพัก รูปแบบ YYYY-MM-DD",
                    },
                    "check_out": {
                        "type": "string",
                        "description": "วันออก รูปแบบ YYYY-MM-DD ต้องเป็นวันหลังวันเข้าพักอย่างน้อย 1 วัน",
                    },
                },
                "required": ["check_in", "check_out"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quote_price",
            "description": (
                "คิดราคาสำหรับการจองใหม่ ใช้ทุกครั้งที่ลูกค้าถามราคารวม "
                "และทุกครั้งที่ลูกค้าบอกจำนวนคน (ใส่ guests) เพื่อให้ระบบคิดว่าต้องจองกี่ห้อง "
                "พักห้องเดียวได้ไหม และต้องขอเตียงเสริมกี่เตียง "
                "ห้ามคูณหรือบวกเลขเอง และห้ามนับจำนวนคืนเอง tool นี้นับให้จากวันที่ "
                "ห้ามใช้ตอบยอดของการจองที่ลูกค้าจองไว้แล้ว ยอดนั้นต้องมาจาก get_my_bookings"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "check_in": {
                        "type": "string",
                        "description": "วันเข้าพัก รูปแบบ YYYY-MM-DD",
                    },
                    "check_out": {
                        "type": "string",
                        "description": "วันออก รูปแบบ YYYY-MM-DD",
                    },
                    "rooms": {
                        "type": "integer",
                        "description": "จำนวนห้องที่จอง ถ้าลูกค้าไม่ได้บอกให้ใส่ 1",
                    },
                    "room_name": {
                        "type": "string",
                        "description": (
                            "ชื่อห้องที่ลูกค้าเจาะจง ต้องตรงกับ name ในผล search_available_rooms "
                            "ถ้าไม่ได้เจาะจงให้เว้นไว้ "
                            "แล้วระบบจะคิดจากห้องที่ถูกที่สุดที่ว่างอยู่"
                        ),
                    },
                    "guests": {
                        "type": "integer",
                        "description": (
                            "จำนวนผู้เข้าพักทั้งหมด (ผู้ใหญ่รวมเด็ก) ใส่เฉพาะเมื่อลูกค้าบอกจำนวนคน "
                            "ในข้อความนี้หรือข้อความก่อนหน้า ห้ามเดา"
                        ),
                    },
                },
                "required": ["check_in", "check_out"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "ค้นข้อมูลรีสอร์ทที่ไม่ได้อยู่ในระบบจอง เช่น ค่าเตียงเสริมและวิธีขอ "
                "ขั้นตอนการชำระเงินโดยรวม นโยบายการยกเลิก เวลาเช็คอิน-เช็คเอาท์ การเดินทาง "
                "สิ่งอำนวยความสะดวกส่วนกลาง สัตว์เลี้ยง และคำถามทั่วไปเกี่ยวกับรีสอร์ท "
                "ห้ามใช้ tool นี้หาราคาห้อง ห้องว่าง ของในห้อง เลขพร้อมเพย์ หรือเวลาที่ต้องชำระ "
                "เพราะคลังความรู้ไม่มีข้อมูลพวกนั้น"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "คำค้นสั้น ๆ ที่สื่อถึงสิ่งที่ลูกค้าอยากรู้",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_payment_info",
            "description": (
                "ดูเลขพร้อมเพย์ (promptPayId) ชื่อบัญชี และระยะเวลาที่ระบบกันห้องไว้ให้ก่อน "
                "ยกเลิกอัตโนมัติ (holdMinutes นาที) ใช้เมื่อลูกค้าถามว่าจ่ายเงินอย่างไร "
                "โอนเข้าเลขอะไร หรือต้องจ่าย/อัปโหลดสลิปภายในกี่นาที"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_restaurants",
            "description": (
                "ดูรายชื่อร้านอาหาร คาเฟ่ และร้านของกินใกล้รีสอร์ท พร้อมประเภทอาหาร เวลาเปิดปิด "
                "และวันหยุดในคำอธิบาย ใช้ทุกครั้งที่ลูกค้าถามเรื่องของกิน ชื่อร้าน หรือร้านเปิดไหม"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_bookings",
            "description": (
                "ดูรายการจองของลูกค้าคนที่ล็อกอินอยู่เท่านั้น (ดูของคนอื่นไม่ได้) "
                "ใช้เมื่อลูกค้าถามถึงการจองของตัวเอง เช่น จองไว้วันไหน ห้องอะไร ยอดเท่าไหร่ "
                "ห้องที่จองมีของอะไร เคยพักกี่ครั้ง ชื่อในการจอง "
                "ผลมี customerName, currentBookings (ยังไม่ถึงหรือกำลังพัก พร้อมรายละเอียดห้อง), "
                "stayedCount (จำนวนครั้งที่เข้าพักจริง), cancelledCount และ recentPastBookings"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_past": {
                        "type": "boolean",
                        "description": (
                            "true เฉพาะเมื่อลูกค้าถามถึงประวัติหรือการจองที่ผ่านไปแล้ว "
                            "เช่น 'เคยจองอะไรบ้าง' 'ประวัติการจอง' 'ครั้งที่แล้ว' "
                            "นอกนั้นไม่ต้องส่ง"
                        ),
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking_payment",
            "description": (
                "ดูสถานะการชำระเงินและการตรวจสลิป ใช้เมื่อลูกค้าถามว่าจ่ายแล้วหรือยัง เงินเข้าหรือยัง "
                "หรือเจ้าหน้าที่ตรวจสลิปหรือยัง ถ้าลูกค้าพิมพ์เลขการจองมา ให้ใส่เลขนั้นตรง ๆ "
                "ถ้าลูกค้าไม่ได้ระบุ (เช่น 'อันนั้นจ่ายแล้วยัง') ให้เว้น booking_id "
                "ระบบจะดูให้ทุกการจองที่ยังไม่ถึง ห้ามเดา booking_id ขึ้นมาเอง"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "เลขการจองที่ลูกค้าพิมพ์มา หรือ id จาก get_my_bookings",
                    }
                },
            },
        },
    },
]


class ToolExecutor:
    """รัน tool ที่โมเดลขอ แล้วคืนผลเป็น JSON string สำหรับใส่กลับเข้า message

    เก็บ hit จาก search_knowledge_base ไว้ใน knowledge_hits ด้วย เพราะ
    ChatResponse.sources ต้องอ้างอิงเอกสารที่ถูกใช้ตอบจริง
    """

    def __init__(
        self,
        booking_client: BookingAPIClient,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        language: str = "th",
        customer_token: str | None = None,
        today: str | None = None,
        user_message: str = "",
        guests_hint: int | None = None,
    ):
        self.booking_client = booking_client
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.language = language
        self.customer_token = customer_token
        #: วันนี้ตามเวลาไทย (YYYY-MM-DD) ใช้แยกการจองปัจจุบันกับประวัติ — ส่งมาจาก
        #: rag_pipeline ให้ตรงกับ {today} ใน SYSTEM_PROMPT
        self.today = today or date.today().isoformat()
        #: ข้อความล่าสุดที่ลูกค้าพิมพ์ ใช้ตรวจว่าปีกับเลขการจองที่โมเดลส่งมาตรงกับที่ลูกค้าพูดจริง
        self.user_message = user_message
        #: จำนวนคนที่ลูกค้าบอกจริง (ข้อความนี้หรือข้อความก่อนหน้า) — rag_pipeline หาให้
        #: ใช้ทิ้ง guests ที่โมเดลแต่งเอง (ทดสอบจริง "2 ห้อง 3 คืน" โมเดลใส่ guests=2 มาเอง
        #: แล้วระบบเลยคิดเป็นห้องเดียว)
        self.guests_hint = guests_hint
        #: ของในห้องของแต่ละห้องที่ระบบจองคืนมาในรอบนี้ — ใช้จับคำตอบที่บอกว่าห้องมีของที่ไม่มี
        self.room_amenities: dict[str, list[str]] = {}
        self.knowledge_hits: list[dict[str, Any]] = []
        self.called_tools: list[str] = []
        #: tool ที่เรียกแล้วได้ข้อมูลกลับมาจริง ไม่ใช่ error — ใช้ตัดสิน ChatResponse.confidence
        #: เพราะ "เรียกแล้ว" กับ "ได้ข้อมูลมา" ไม่เหมือนกัน เช่น ถาม booking ตอนยังไม่ล็อกอิน
        self.successful_tools: list[str] = []
        #: ผลของ tool ทุกตัวในรอบนี้ (JSON string) ใช้ตรวจว่าลิงก์ในคำตอบมาจากแหล่งจริงไหม
        self.tool_outputs: list[str] = []
        #: จำนวนคนที่ส่งเข้า quote_price ในรอบนี้ — rag_pipeline ใช้ตัดสินว่าต้องกระทุ้งไหม
        self.planned_guests: list[int] = []
        self.date_error_seen = False
        #: ชื่อห้องทุกห้องที่ระบบจองเคยคืนมาในรอบนี้ — rag_pipeline ใช้จับชื่อห้องที่โมเดลแต่งขึ้น
        #: (ทดสอบจริงตอบว่า "ห้องที่ไม่ว่างคือ P5 และ P6" ทั้งที่ไม่มีห้องชื่อนี้)
        self.known_room_names: set[str] = set()
        #: ห้องของการจองที่ยังไม่ถึงของลูกค้าคนนี้ — ใช้จับคำตอบ "ห้อง P1 ที่คุณจองไว้" ทั้งที่จอง F1
        self.booked_room_names: set[str] = set()
        self.extra_bed_rate_checked = False
        self._extra_bed_rate: int | None = None
        self._room_cache: dict[str, dict[str, Any]] = {}

    async def run(self, name: str, arguments: dict[str, Any]) -> str:
        self.called_tools.append(name)
        try:
            result = await self._dispatch(name, arguments)
        except BookingAPIAuthError:
            result = _LOGIN_REQUIRED
        except BookingAPIError as exc:
            # ส่ง error กลับไปเป็นผลของ tool แทนที่จะโยนขึ้นไป เพื่อให้โมเดลบอกลูกค้าได้ว่า
            # ตอนนี้ดูข้อมูลให้ไม่ได้ ดีกว่าปล่อยให้ทั้ง request พังเป็น 500
            result = {"error": "booking_api_unavailable", "detail": str(exc)}
        except (KeyError, TypeError, ValueError) as exc:
            result = {"error": "bad_arguments", "detail": str(exc)}
        is_error = isinstance(result, dict) and "error" in result
        if not is_error:
            self.successful_tools.append(name)
        elif result["error"] in _DATE_ERRORS:
            self.date_error_seen = True
        output = json.dumps(result, ensure_ascii=False)
        self.tool_outputs.append(output)
        return output

    def _check_dates(self, check_in: str, check_out: str) -> dict[str, Any] | None:
        invalid = _check_stay_dates(check_in, check_out, self.today)
        if invalid:
            return invalid
        # ทดสอบจริง: ลูกค้าพิมพ์ "29 ตุลา 2004" แต่โมเดลส่ง 2026-10-29 มาเอง (เปลี่ยนปีก่อนเรียก tool)
        # การเช็ควันในอดีตจึงจับไม่ได้ ต้องเทียบกับปีที่ลูกค้าพิมพ์จริง
        start = date.fromisoformat(check_in)
        years = years_mentioned(self.user_message, this_year=int(self.today[:4]))
        if years and start.year not in years:
            return {
                "error": "year_mismatch",
                "detail": f"ลูกค้าระบุปี {sorted(years)} แต่ส่งปี {start.year} มา",
                "hint": "ส่งปีตามที่ลูกค้าระบุ (แปลง พ.ศ. เป็น ค.ศ.) ห้ามเปลี่ยนปีเอง แล้วเรียก tool ใหม่",
            }
        slash_dates = _slash_dates(self.user_message)
        if slash_dates and (start.day, start.month) not in slash_dates:
            day, month = slash_dates[0]
            today_date = date.fromisoformat(self.today)
            try:
                suggested = date(today_date.year, month, day)
                if suggested < today_date:
                    suggested = date(today_date.year + 1, month, day)
            except ValueError:
                suggested = None
            return {
                "error": "date_format_mismatch",
                "detail": f"ลูกค้าพิมพ์ {day}/{month} ซึ่งแบบไทยคือวันที่ {day} เดือนที่ {month}",
                # รอบแรก hint บอกแค่หลักการ โมเดลเลยไปขอให้ลูกค้าพิมพ์ใหม่เป็น YYYY-MM-DD
                "hint": (
                    f"ห้ามถามลูกค้ากลับ ให้เรียก tool ใหม่ทันทีด้วย check_in = {suggested.isoformat()}"
                    if suggested else "วันที่นี้ไม่มีจริง ให้บอกลูกค้าแล้วถามวันที่ใหม่"
                ),
            }
        # ทดสอบจริง "อยู่ 2 อาทิตย์ ราคาเท่าไหร่" โมเดลส่งช่วง 2 คืนมา แล้วตอบราคา 2 คืน
        nights = (date.fromisoformat(check_out) - start).days
        wanted_nights = nights_mentioned(self.user_message)
        if wanted_nights and nights != wanted_nights:
            return {
                "error": "nights_mismatch",
                "detail": f"ลูกค้าบอกว่าพัก {wanted_nights} คืน แต่ช่วงวันที่ที่ส่งมาคือ {nights} คืน",
                "hint": (
                    f"ให้ check_out = check_in + {wanted_nights} วัน "
                    f"เช่น check_in {check_in} ต้องเป็น check_out "
                    f"{(start + timedelta(days=wanted_nights)).isoformat()} แล้วเรียก tool ใหม่"
                ),
            }
        # "เมื่อวาน" ไม่อยู่ในปฏิทินอนาคต โมเดลเลยส่งวันนี้มาแล้วตอบว่าเมื่อวานมีห้องว่าง
        if _YESTERDAY_RE.search(self.user_message) and check_in == self.today:
            return {
                "error": "date_in_past",
                "detail": "ลูกค้าถามถึงเมื่อวาน ซึ่งผ่านมาแล้ว จองย้อนหลังไม่ได้",
                "hint": "บอกลูกค้าว่าวันที่ผ่านมาแล้วจองไม่ได้ แล้วถามว่าต้องการวันไหน",
            }
        # หลังเจอ error เรื่องวันที่ โมเดลชอบหนีไปค้น "วันนี้" แทน (ตามกติกาข้อ 11 ที่ใช้ตอนไม่มีวันที่)
        # แล้วตอบห้องว่างของวันนี้ราวกับเป็นวันที่ลูกค้าถาม
        if self.date_error_seen and check_in == self.today and not _mentions_today(self.user_message):
            return {
                "error": "must_ask_customer",
                "detail": "ลูกค้าระบุวันที่มาแล้วแต่ใช้ไม่ได้ ห้ามเปลี่ยนไปค้นวันนี้แทน",
                "hint": "ให้บอกลูกค้าตามปัญหาเรื่องวันที่ที่เจอ แล้วถามว่าต้องการวันไหน",
            }
        return None

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        if name in ("search_available_rooms", "quote_price"):
            invalid = self._check_dates(arguments["check_in"], arguments["check_out"])
            if invalid:
                return invalid
        if name == "search_available_rooms":
            return await self._search_rooms(arguments["check_in"], arguments["check_out"])
        if name == "quote_price":
            guests = self.guests_hint if arguments.get("guests") else None
            if guests and self.customer_token and _EXISTING_BOOKING_RE.search(self.user_message):
                # ทดสอบจริงสองรอบ: "การจองที่มีอยู่ ขอเพิ่มเป็น 3 คน" ได้ห้องใหม่ (P1) มาแทนห้องเดิม (F1)
                return {
                    "error": "existing_booking",
                    "detail": "ลูกค้าถามถึงการเพิ่มคนในการจองที่มีอยู่ ไม่ใช่การจองใหม่",
                    "hint": (
                        "เรียก get_my_bookings ดูห้องเดิม แล้วบอกว่าเพิ่มได้ด้วยการโทรขอเตียงเสริม"
                        "สำหรับห้องเดิมที่ 081-598-1199 (ห้องละ 1 เตียง) "
                        "ค่าเตียงเสริมดูจาก search_knowledge_base"
                    ),
                }
            if guests:
                return await self._plan_for_guests(arguments["check_in"], arguments["check_out"], guests)
            # ไม่อ่าน extra_beds / extra_bed_price_per_night โดยตั้งใจ ถ้าโมเดลยังส่งมา
            # (จำ schema เก่า) ให้เพิกเฉย ไม่ใช่ error — ยอดต้องเป็นค่าห้องอย่างเดียวเสมอ
            rooms = int(arguments.get("rooms") or 1)
            if rooms == 1:
                rooms = rooms_mentioned(self.user_message) or 1
            quote = await self.booking_client.quote_stay(
                check_in=arguments["check_in"],
                check_out=arguments["check_out"],
                rooms=rooms,
                room_name=arguments.get("room_name"),
            )
            self.known_room_names.add(quote["roomName"])
            quote["dateRangeLabel"] = format_stay_dates(
                arguments["check_in"], arguments["check_out"], self.language
            )
            return quote
        if name == "search_knowledge_base":
            return self._search_knowledge_base(arguments["query"])
        if name == "get_payment_info":
            return await self.booking_client.get_payment_info()
        if name == "list_restaurants":
            return await self.booking_client.list_restaurants()
        if name == "get_my_bookings":
            if not self.customer_token:
                return _LOGIN_REQUIRED
            return await self._my_bookings(include_past=bool(arguments.get("include_past")))
        if name == "get_booking_payment":
            if not self.customer_token:
                return _LOGIN_REQUIRED
            return await self._booking_payment(arguments.get("booking_id"))
        return {"error": "unknown_tool", "detail": name}

    async def _search_rooms(self, check_in: str, check_out: str) -> dict[str, Any]:
        found = await self.booking_client.search_rooms(check_in=check_in, check_out=check_out)
        rooms = [self._localize_room(room) for room in found]
        available = {room["name"] for room in rooms}
        try:
            listed = await self.booking_client.list_rooms()
        except BookingAPIError:
            listed = []
        if not isinstance(listed, list):
            listed = []
        self.known_room_names |= available | {room["name"] for room in listed}
        # กติกาเรื่องเตียงเสริมอยู่ใน SYSTEM_PROMPT แล้ว แต่ทดสอบจริงโมเดลยังข้ามการค้นคลังความรู้
        # แล้วแต่งค่าเตียงเสริมขึ้นเอง ("คืนละ 300 บาท" ทั้งที่จริงคือ 200) ติด note ไว้กับผล tool
        # เพราะโมเดลยึดสิ่งที่อยู่ในผล tool มากกว่าร้อยแก้วใน prompt
        # ตัวเลขยังไม่ใส่ในโค้ดโดยตั้งใจ — คลังความรู้เป็นเจ้าของค่าเตียงเสริม
        return {
            "dateRangeLabel": format_stay_dates(check_in, check_out, self.language),
            "rooms": rooms,
            "unavailableRoomNames": sorted(
                room["name"] for room in listed if room["name"] not in available
            ),
            "note": _ROOMS_NOTE,
        }

    async def _plan_for_guests(self, check_in: str, check_out: str, guests: int) -> dict[str, Any]:
        if guests < 1:
            return {"error": "bad_arguments", "detail": "guests ต้องเป็น 1 ขึ้นไป"}
        self.planned_guests.append(guests)
        available = await self.booking_client.search_rooms(check_in=check_in, check_out=check_out)
        nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
        plan = plan_rooms_for_guests(
            available,
            guests=guests,
            nights=nights,
            extra_bed_rate=self._lookup_extra_bed_rate(),
            max_extra_beds_per_room=_MAX_EXTRA_BEDS_PER_ROOM,
        )
        self.known_room_names |= {room["name"] for room in available}
        plan["dateRangeLabel"] = format_stay_dates(check_in, check_out, self.language)
        plan["note"] = (
            "ตอบตาม options เท่านั้น ห้ามคิดเลขเอง ถ้ามีสองแบบให้เสนอทั้งสองแบบ "
            "fitsInOneRoom / fitsInOneRoomWithExtraBed บอกว่าพักห้องเดียวได้ไหม "
            "guestsToEnterInApp คือจำนวนคนที่กรอกในแอป เตียงเสริมโทรขอที่ 081-598-1199 "
            "payInApp ชำระในแอป payAtResortForExtraBeds ชำระที่รีสอร์ทวันเช็คอิน "
            "ถ้าลูกค้าถามยอดรวมทั้งหมดให้ใช้ grandTotal และแยกบอกสองส่วนด้วย "
            "ถ้า extraBedRatePerNight เป็น null แปลว่าไม่มีข้อมูลค่าเตียงเสริม ห้ามเดา"
        )
        return plan

    def _lookup_extra_bed_rate(self) -> int | None:
        """อ่านค่าเตียงเสริมจากคลังความรู้ (ภาษาไทยเสมอ เพราะ regex อ่านจากข้อความไทย)"""
        if self.extra_bed_rate_checked:
            return self._extra_bed_rate
        self.extra_bed_rate_checked = True
        hits = self.vector_store.query(
            self.embedding_service.embed_query(_EXTRA_BED_QUERY),
            top_k=settings.retrieval_top_k,
            language="th",
        )
        for hit in hits if isinstance(hits, list) else []:
            match = _EXTRA_BED_RATE_RE.search(hit.get("text", ""))
            if match:
                self._extra_bed_rate = int(match.group(1).replace(",", ""))
                self.knowledge_hits.append(hit)
                break
        return self._extra_bed_rate

    def known_extra_bed_rate(self) -> int | None:
        """ค่าเตียงเสริมจากผลค้นคลังความรู้ในรอบนี้ (ทั้งที่โมเดลค้นเองและที่ quote_price ค้นให้)"""
        if self._extra_bed_rate is not None:
            return self._extra_bed_rate
        for hit in self.knowledge_hits:
            match = _EXTRA_BED_RATE_RE.search(hit.get("text", ""))
            if match:
                return int(match.group(1).replace(",", ""))
        return None

    async def _my_bookings(self, include_past: bool) -> dict[str, Any]:
        bookings = await self.booking_client.get_my_bookings(self.customer_token)
        result = group_bookings(bookings, today=self.today, include_past=include_past)
        # ทดสอบจริง: "ห้องที่จองไว้มีตู้เย็นไหม" บอทถามวันที่กลับ และ "ห้องที่จองไว้เตียงอะไร"
        # บอทไปค้นห้องว่างแล้วหยิบชื่อห้องผิด (ตอบ P1 ทั้งที่จอง F1) ใส่รายละเอียดห้องให้เลย
        for booking in result["currentBookings"]:
            self.booked_room_names.add(booking.get("roomName", ""))
            room = await self._room_details(booking.get("roomId", ""))
            if room:
                booking["roomTypeLabel"] = room["typeLabel"]
                booking["roomAmenities"] = room["amenities"]
        for key in ("currentBookings", "recentPastBookings", "pastBookings"):
            for booking in result.get(key, []):
                booking.pop("roomId", None)
                self.known_room_names.add(booking.get("roomName", ""))
                booking["dateRangeLabel"] = format_stay_dates(
                    str(booking["checkIn"])[:10], str(booking["checkOut"])[:10], self.language
                )
        # ทดสอบจริง "ยังมีอันไหนที่ยังไม่ได้จ่าย" บอทยกรายการที่ยกเลิกไปแล้ว (unpaid) มาบอกว่าค้างจ่าย
        result["unpaidUpcomingBookings"] = [
            b["id"] for b in result["currentBookings"] if b.get("paymentStatus") == "unpaid"
        ]
        result["note"] = (
            "ข้อมูลนี้เป็นของลูกค้าที่ล็อกอินอยู่ (customerName) เท่านั้น ถ้าลูกค้าถามถึงการจองของ"
            "คนอื่นหรืออีเมลอื่น ให้บอกว่าดูได้เฉพาะการจองของตัวเอง ห้ามบอกว่าข้อมูลนี้เป็นของคนนั้น "
            "ยอดของการจองที่มีอยู่ใช้ totalPrice ห้ามเรียก quote_price "
            "เคยเข้าพักกี่ครั้งใช้ stayedCount (pastBookingsCount รวมรายการที่ยกเลิกด้วย) "
            "การจองที่ยังต้องจ่ายมีเฉพาะใน unpaidUpcomingBookings ถ้าว่างแปลว่าไม่มีอะไรค้างจ่าย "
            "รายการที่ status เป็น cancelled ไม่ต้องจ่ายแล้ว ห้ามบอกว่าค้างจ่าย"
        )
        return result

    async def _room_details(self, room_id: str) -> dict[str, Any] | None:
        if not room_id:
            return None
        if room_id not in self._room_cache:
            try:
                room = await self.booking_client.get_room(room_id)
            except BookingAPIError:
                return None
            self._room_cache[room_id] = self._localize_room(room)
        return self._room_cache[room_id]

    async def _booking_payment(self, booking_id: str | None) -> Any:
        """สถานะการชำระเงิน — ตรวจก่อนเสมอว่าการจองนั้นเป็นของลูกค้าคนนี้

        ทดสอบจริงเจอสองแบบ:
        - ลูกค้าพิมพ์เลขการจอง 2e76d255… แต่โมเดลส่ง id ของอีกรายการมา แล้วตอบสถานะผิดรายการ
          จึงใช้เลขที่ลูกค้าพิมพ์เองแทนเลขที่โมเดลส่ง
        - "การจองของฉัน" -> "อันนั้นจ่ายแล้วยัง" โมเดลไม่มี id ในประวัติ เลยแต่ง id ขึ้นมา
          แล้วได้ "ไม่มีข้อมูล" จึงให้เว้น booking_id ได้ แล้วดูทุกการจองที่ยังไม่ถึงให้
        การเช็คความเป็นเจ้าของฝั่งนี้ทำให้ id ของคนอื่นได้คำตอบว่า "ไม่พบ" แทน
        "ต้องล็อกอิน" (backend ตอบ 403 ซึ่งถูกแปลงเป็น not_authenticated)
        """
        bookings = await self.booking_client.get_my_bookings(self.customer_token)
        mentioned = [m.lower() for m in _UUID_RE.findall(self.user_message)]
        if mentioned:
            booking_id = mentioned[0]
        if booking_id:
            wanted = booking_id.strip().lower()
            targets = [b for b in bookings if b["id"].lower() == wanted] or [
                b for b in bookings if len(wanted) >= 8 and b["id"].lower().startswith(wanted)
            ]
            if not targets:
                return {
                    "error": "booking_not_found",
                    "detail": "ไม่พบการจองเลขนี้ในบัญชีของลูกค้า",
                    "hint": (
                        "บอกลูกค้าว่าไม่พบการจองเลขนี้ในบัญชีที่ล็อกอินอยู่ "
                        "ห้ามเดาเลขอื่นแทน และห้ามบอกว่าเป็นการจองของคนอื่น"
                    ),
                }
        else:
            targets = group_bookings(bookings, today=self.today)["currentBookings"]
            if not targets:
                return {"bookings": [], "note": "ลูกค้าไม่มีการจองที่ยังไม่ถึง"}

        results = []
        for booking in targets:
            try:
                payment = await self.booking_client.get_booking_payment(
                    booking_id=booking["id"], token=self.customer_token
                )
            except BookingAPIAuthError:
                raise
            except BookingAPIError:
                payment = None  # ยังไม่มีรายการชำระเงิน เช่นยังไม่ได้อัปโหลดสลิป
            summary = {k: v for k, v in booking.items() if k not in ("customerName", "roomId")}
            summary["dateRangeLabel"] = format_stay_dates(
                str(booking["checkIn"])[:10], str(booking["checkOut"])[:10], self.language
            )
            results.append({"booking": summary, "payment": payment})
        return {"bookings": results}

    def _localize_room(self, room: dict[str, Any]) -> dict[str, Any]:
        room = dict(room)
        self.room_amenities[room.get("name", "")] = list(room.get("amenities", []))
        if self.language == "th":
            room["amenities"] = [_AMENITY_LABELS_TH.get(a, a) for a in room.get("amenities", [])]
        return room

    def _search_knowledge_base(self, query: str) -> list[dict[str, Any]]:
        """ค้นด้วยคำค้นของโมเดล และค้นด้วยคำถามของลูกค้าตรง ๆ อีกรอบ แล้วรวมผล

        ทดสอบจริง: โมเดลย่อ "มีสระว่ายน้ำไหม" เหลือคำค้น "สระว่ายน้ำ" ผลคือ 5 อันดับแรก
        ไม่มีเอกสารเรื่องสระเลย ขณะที่ค้นด้วยคำถามเต็มได้ faq กับ facilities ที่ตอบได้
        เป็นอันดับ 1-2 (คลังความรู้เขียนเป็นรูป ถาม/ตอบ จึงเข้ากับประโยคคำถามมากกว่าคำโดด ๆ)
        """
        queries = [query]
        if self.user_message and self.user_message.strip() != query.strip():
            queries.append(self.user_message)
        merged: dict[str, dict[str, Any]] = {}
        for text in queries:
            hits = self.vector_store.query(
                self.embedding_service.embed_query(text),
                top_k=settings.retrieval_top_k,
                language=self.language,
            )
            for hit in hits if isinstance(hits, list) else []:
                kept = merged.get(hit["doc_id"])
                if kept is None or (hit["distance"] or 1) < (kept["distance"] or 1):
                    merged[hit["doc_id"]] = hit
        hits = sorted(merged.values(), key=lambda h: h["distance"] if h["distance"] is not None else 1)
        hits = hits[: settings.retrieval_top_k]
        self.knowledge_hits.extend(hits)
        return [{"text": hit["text"]} for hit in hits]


_TODAY_WORDS = ("วันนี้", "คืนนี้", "today", "tonight")
_EXISTING_BOOKING_RE = re.compile(r"ที่มีอยู่|ที่จองไว้|การจองเดิม|ห้องเดิม|การจองของฉัน|my booking", re.I)


def _mentions_today(message: str) -> bool:
    lowered = message.lower()
    return any(word in lowered for word in _TODAY_WORDS)
