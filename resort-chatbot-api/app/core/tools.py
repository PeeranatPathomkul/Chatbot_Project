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
from datetime import date
from typing import Any

from app.config import settings
from app.services.booking_api import (
    BookingAPIAuthError,
    BookingAPIClient,
    BookingAPIError,
    group_bookings,
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

#: ผลที่ส่งกลับให้โมเดลเมื่อ tool ต้องใช้สิทธิ์แต่ลูกค้ายังไม่ได้ล็อกอิน
_LOGIN_REQUIRED = {
    "error": "not_authenticated",
    "hint": "ลูกค้ายังไม่ได้ล็อกอิน ให้แจ้งลูกค้าว่าต้องเข้าสู่ระบบก่อนจึงจะดูข้อมูลการจองของตัวเองได้",
}

#: แนบไปกับผลของ search_available_rooms ทุกครั้ง (เหตุผลอยู่ที่ ToolExecutor._dispatch)
_ROOMS_NOTE = (
    "capacity คือจำนวนผู้เข้าพักสูงสุดที่จองในแอปได้ต่อห้อง ถ้าลูกค้ามาเกิน capacity "
    "หรือถามเรื่องเตียงเสริม ต้องเรียก search_knowledge_base ค้น 'ค่าเตียงเสริม' ก่อนตอบ "
    "ผลนี้ไม่มีค่าเตียงเสริม ห้ามเดาตัวเลข และห้ามแสดงชื่อ field ให้ลูกค้าเห็น"
)

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
                "คิดยอดรวมค่าห้องที่ต้องชำระในแอปให้ ใช้ tool นี้ทุกครั้งที่ลูกค้าถามราคารวม "
                "ห้ามคูณหรือบวกเลขเอง และห้ามนับจำนวนคืนเอง tool นี้นับให้จากวันที่ "
                "ยอดที่ได้เป็นค่าห้องอย่างเดียว ไม่รวมเตียงเสริม "
                "(เตียงเสริมชำระแยกที่รีสอร์ทวันเช็คอิน ห้ามบวกเข้ายอดนี้)"
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
            "description": "ดูรายชื่อร้านอาหารที่รีสอร์ทแนะนำ ใช้เมื่อลูกค้าถามเรื่องที่กินข้าว",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_bookings",
            "description": (
                "ดูรายการจองของลูกค้าคนที่กำลังคุยอยู่ ใช้เมื่อลูกค้าถามถึงการจองของตัวเอง "
                "เช่น จองไว้วันไหน สถานะเป็นอย่างไร ยอดเท่าไหร่ "
                "ผลลัพธ์ปกติมีเฉพาะการจองที่ยังไม่ถึงหรือกำลังพักอยู่ (currentBookings) "
                "และจำนวนรายการที่ผ่านไปแล้ว (pastBookingsCount)"
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
                "ดูสถานะการชำระเงินของการจองรายการหนึ่ง ใช้เมื่อลูกค้าถามว่าเงินเข้าหรือยัง "
                "ต้องได้ booking_id จาก get_my_bookings ก่อน ห้ามเดา booking_id ขึ้นมาเอง"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_id": {
                        "type": "string",
                        "description": "id ของการจองที่ได้จาก get_my_bookings",
                    }
                },
                "required": ["booking_id"],
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
    ):
        self.booking_client = booking_client
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.language = language
        self.customer_token = customer_token
        #: วันนี้ตามเวลาไทย (YYYY-MM-DD) ใช้แยกการจองปัจจุบันกับประวัติ — ส่งมาจาก
        #: rag_pipeline ให้ตรงกับ {today} ใน SYSTEM_PROMPT
        self.today = today or date.today().isoformat()
        self.knowledge_hits: list[dict[str, Any]] = []
        self.called_tools: list[str] = []
        #: tool ที่เรียกแล้วได้ข้อมูลกลับมาจริง ไม่ใช่ error — ใช้ตัดสิน ChatResponse.confidence
        #: เพราะ "เรียกแล้ว" กับ "ได้ข้อมูลมา" ไม่เหมือนกัน เช่น ถาม booking ตอนยังไม่ล็อกอิน
        self.successful_tools: list[str] = []

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
        if not (isinstance(result, dict) and "error" in result):
            self.successful_tools.append(name)
        return json.dumps(result, ensure_ascii=False)

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "search_available_rooms":
            rooms = await self.booking_client.search_rooms(
                check_in=arguments["check_in"], check_out=arguments["check_out"]
            )
            # กติกาเรื่องเตียงเสริมอยู่ใน SYSTEM_PROMPT แล้ว แต่ทดสอบจริงโมเดลยังข้ามการค้นคลังความรู้
            # แล้วแต่งค่าเตียงเสริมขึ้นเอง ("คืนละ 300 บาท" ทั้งที่จริงคือ 200) ติด note ไว้กับผล tool
            # เพราะโมเดลยึดสิ่งที่อยู่ในผล tool มากกว่าร้อยแก้วใน prompt
            # ตัวเลขยังไม่ใส่ในโค้ดโดยตั้งใจ — คลังความรู้เป็นเจ้าของค่าเตียงเสริม
            return {"rooms": rooms, "note": _ROOMS_NOTE}
        if name == "quote_price":
            # ไม่อ่าน extra_beds / extra_bed_price_per_night โดยตั้งใจ ถ้าโมเดลยังส่งมา
            # (จำ schema เก่า) ให้เพิกเฉย ไม่ใช่ error — ยอดต้องเป็นค่าห้องอย่างเดียวเสมอ
            return await self.booking_client.quote_stay(
                check_in=arguments["check_in"],
                check_out=arguments["check_out"],
                rooms=int(arguments.get("rooms") or 1),
                room_name=arguments.get("room_name"),
            )
        if name == "search_knowledge_base":
            return self._search_knowledge_base(arguments["query"])
        if name == "get_payment_info":
            return await self.booking_client.get_payment_info()
        if name == "list_restaurants":
            return await self.booking_client.list_restaurants()
        if name == "get_my_bookings":
            if not self.customer_token:
                return _LOGIN_REQUIRED
            bookings = await self.booking_client.get_my_bookings(self.customer_token)
            return group_bookings(
                bookings,
                today=self.today,
                include_past=bool(arguments.get("include_past")),
            )
        if name == "get_booking_payment":
            if not self.customer_token:
                return _LOGIN_REQUIRED
            return await self.booking_client.get_booking_payment(
                booking_id=arguments["booking_id"], token=self.customer_token
            )
        return {"error": "unknown_tool", "detail": name}

    def _search_knowledge_base(self, query: str) -> list[dict[str, Any]]:
        embedding = self.embedding_service.embed_query(query)
        hits = self.vector_store.query(
            embedding, top_k=settings.retrieval_top_k, language=self.language
        )
        self.knowledge_hits.extend(hits)
        return [{"text": hit["text"]} for hit in hits]
