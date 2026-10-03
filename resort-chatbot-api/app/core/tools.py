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
  ห้องว่าง ราคาห้อง ชื่อห้อง capacity สิ่งอำนวยความสะดวกในห้อง
- คลังความรู้ (Chroma) เป็นเจ้าของ:
  ค่าเตียงเสริม มัดจำ นโยบายยกเลิก เวลาเช็คอิน สิ่งอำนวยความสะดวกส่วนกลาง การเดินทาง
ตัวเลขที่ระบบจองเป็นเจ้าของถูกถอดออกจากคลังความรู้แล้ว เพื่อไม่ให้มีสองแหล่งที่ขัดกันเอง
(ถ้าเติมราคากลับเข้าไปใน data/*.md เมื่อไหร่ บอทจะเริ่มตอบราคาที่ไม่ตรงกับที่จองจริงได้)

**customer_token ต้องไม่ใช่ parameter ที่โมเดลกรอก**
มันถูกฉีดเข้ามาตอนสร้าง ToolExecutor จาก header ของ request ถ้าปล่อยให้เป็น
parameter ใน schema โมเดลจะแต่ง token ขึ้นมาเองได้ และคำสั่งที่แฝงมาในข้อความลูกค้า
ก็สั่งให้มันใช้ token ของคนอื่นได้ด้วย
"""

import json
from typing import Any

from app.config import settings
from app.services.booking_api import (
    BookingAPIAuthError,
    BookingAPIClient,
    BookingAPIError,
)
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore

#: ผลที่ส่งกลับให้โมเดลเมื่อ tool ต้องใช้สิทธิ์แต่ลูกค้ายังไม่ได้ล็อกอิน
_LOGIN_REQUIRED = {
    "error": "not_authenticated",
    "hint": "ลูกค้ายังไม่ได้ล็อกอิน ให้แจ้งลูกค้าว่าต้องเข้าสู่ระบบก่อนจึงจะดูข้อมูลการจองของตัวเองได้",
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_available_rooms",
            "description": (
                "ค้นหาห้องพักที่ว่างจริงในช่วงวันที่ที่ระบุ พร้อมราคาต่อคืนและจำนวนผู้เข้าพักที่รับได้ "
                "แต่ละห้องคืนมาสามค่าที่ต้องใช้: capacity คือจำนวนคนที่นอนได้ด้วยเตียงที่มีอยู่แล้ว, "
                "maxGuestsWithExtraBed คือจำนวนสูงสุดที่ห้องรับได้เมื่อขอเตียงเสริม, "
                "และ typeLabel คือชนิดเตียงเป็นภาษาไทย ใช้กรองเมื่อลูกค้าระบุว่าอยากได้เตียงแบบไหน "
                "ใช้ tool นี้ทุกครั้งที่ลูกค้าถามเรื่องห้องว่าง ราคาห้อง หรือขอให้คิดราคารวม "
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
                "คิดยอดรวมค่าที่พักให้ ใช้ tool นี้ทุกครั้งที่ลูกค้าถามราคารวม "
                "ห้ามคูณหรือบวกเลขเอง และห้ามนับจำนวนคืนเอง tool นี้นับให้จากวันที่ "
                "ถ้าลูกค้ายังไม่ได้บอกจำนวนผู้เข้าพัก ให้ใส่ extra_beds เป็น 0 "
                "แล้วค่อยถามลูกค้าว่ามากันกี่ท่าน ห้ามเดาเอง"
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
                    "extra_beds": {
                        "type": "integer",
                        "description": (
                            "จำนวนเตียงเสริมทั้งหมด ใส่มากกว่า 0 ได้เฉพาะเมื่อลูกค้าขอเตียงเสริม "
                            "หรือบอกจำนวนผู้เข้าพักที่เกิน capacity ของห้องเท่านั้น "
                            "ถ้าลูกค้าไม่ได้บอกจำนวนคน ต้องใส่ 0"
                        ),
                    },
                    "extra_bed_price_per_night": {
                        "type": "number",
                        "description": (
                            "อัตราค่าเตียงเสริมต่อเตียงต่อคืน ต้องเอามาจากผล "
                            "search_knowledge_base เท่านั้น ห้ามเดา "
                            "จำเป็นต้องใส่เมื่อ extra_beds มากกว่า 0"
                        ),
                    },
                    "room_name": {
                        "type": "string",
                        "description": (
                            "ชื่อห้องที่ลูกค้าเจาะจง เช่น P1 ถ้าไม่ได้เจาะจงให้เว้นไว้ "
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
                "ค้นข้อมูลรีสอร์ทที่ไม่ได้อยู่ในระบบจอง เช่น ค่าเตียงเสริม เงินมัดจำ "
                "นโยบายการยกเลิก เวลาเช็คอิน-เช็คเอาท์ การเดินทาง สิ่งอำนวยความสะดวกส่วนกลาง "
                "สัตว์เลี้ยง และคำถามทั่วไปเกี่ยวกับรีสอร์ท "
                "ห้ามใช้ tool นี้หาราคาห้องหรือห้องว่าง เพราะคลังความรู้ไม่มีข้อมูลสองอย่างนั้น"
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
                "ดูวิธีชำระเงินและระยะเวลาที่ระบบกันห้องไว้ให้ก่อนยกเลิกอัตโนมัติ "
                "ใช้เมื่อลูกค้าถามว่าจ่ายเงินอย่างไร หรือต้องจ่ายภายในกี่นาที"
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
                "เช่น จองไว้วันไหน สถานะเป็นอย่างไร ยอดเท่าไหร่"
            ),
            "parameters": {"type": "object", "properties": {}},
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
    ):
        self.booking_client = booking_client
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.language = language
        self.customer_token = customer_token
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
            return await self.booking_client.search_rooms(
                check_in=arguments["check_in"], check_out=arguments["check_out"]
            )
        if name == "quote_price":
            return await self.booking_client.quote_stay(
                check_in=arguments["check_in"],
                check_out=arguments["check_out"],
                rooms=int(arguments.get("rooms") or 1),
                extra_beds=int(arguments.get("extra_beds") or 0),
                extra_bed_price_per_night=arguments.get("extra_bed_price_per_night"),
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
            return await self.booking_client.get_my_bookings(self.customer_token)
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
