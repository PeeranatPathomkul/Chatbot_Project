"""
Client สำหรับเรียก Booking API (NestJS backend ที่เป็นเจ้าของ PostgreSQL)

**ทำไมไม่ต่อ PostgreSQL ตรง ๆ** — ดูเหตุผลเต็มใน app/config.py ที่ booking_api_base_url
สรุปสั้น ๆ: ข้อมูลที่แชทบอทต้องใช้ที่สุด ("ห้องนี้ว่างไหม") ไม่ได้ถูกเก็บไว้ในตาราง
แต่เป็นผลลัพธ์ที่ API คำนวณจาก booking ที่มีอยู่ การ query เองคือการเขียนกติกาชุดนั้นซ้ำ

**เรื่องสิทธิ์** — endpoint ที่ขึ้นต้นด้วย `me` หรือรับ booking_id จะรู้ว่าลูกค้าคือใคร
จาก JWT เท่านั้น ไม่มีทางส่ง customer_id เข้าไปถามแทนคนอื่นได้ เราจึงต้องส่ง token
ของลูกค้าคนที่กำลังคุยอยู่เข้ามาทุกครั้ง และ **ห้ามมี token กลางของระบบ** ที่อ่าน
booking ได้ทุกคนเด็ดขาด

**เรื่อง token หมดอายุ** — access token อายุ 15 นาที ซึ่งสั้นกว่าบทสนทนาได้ง่าย ๆ
การ refresh ต้องใช้ refresh token ที่อยู่กับแอปฝั่งลูกค้า ไม่ได้อยู่กับแชทบอท
เจอ 401 เราจึงไม่ refresh เอง แต่โยน BookingAPIAuthError ให้ชั้นบนบอกลูกค้า
ให้ล็อกอินใหม่ (แอปเป็นคนจัดการ refresh + ยิงคำถามเดิมซ้ำ)

**เรื่อง rate limit** — NestJS จำกัด 120 ครั้ง/นาที/IP และคำขอของลูกค้าทุกคนออกจาก IP
เดียวของบอท เราจึงแนบ `X-Internal-Key` ทุกคำขอเพื่อให้ NestJS ไม่นับ rate limit (เฉพาะ GET)
key นี้ไม่ได้ให้สิทธิ์อ่านข้อมูลเพิ่ม จึงไม่ขัดกับหลัก "ห้ามมี token กลาง" ข้างบน
และด้วยเหตุนี้ client ตัวนี้จึงเรียกได้แค่ GET เท่านั้น
"""

from datetime import date
from functools import lru_cache
from typing import Any

import httpx

from app.config import settings


class BookingAPIError(Exception):
    """เรียก Booking API ไม่สำเร็จด้วยเหตุผลที่ไม่ใช่เรื่องสิทธิ์"""


class BookingAPIAuthError(BookingAPIError):
    """ไม่มี token หรือ token หมดอายุ — ต้องให้ลูกค้าล็อกอินใหม่"""


class BookingAPIClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        internal_key: str | None = None,
    ):
        self.base_url = (base_url or settings.booking_api_base_url).rstrip("/")
        self.timeout = timeout or settings.booking_api_timeout_seconds
        self._internal_key = (
            settings.booking_api_internal_key if internal_key is None else internal_key
        )

    def _headers(self, token: str | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._internal_key:
            headers["X-Internal-Key"] = self._internal_key
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> Any:
        headers = self._headers(token)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    f"{self.base_url}{path}", params=params, headers=headers
                )
            except httpx.RequestError as exc:
                raise BookingAPIError(f"ติดต่อระบบจองไม่ได้: {exc}") from exc

        if response.status_code in (401, 403):
            # 403 = booking นี้เป็นของลูกค้าคนอื่น ห้ามเปิดเผยว่ามีอยู่จริง
            # จึงกลืนรวมกับ 401 แล้วให้ชั้นบนตอบแบบ "ไม่พบ/ต้องล็อกอินใหม่"
            raise BookingAPIAuthError("ไม่มีสิทธิ์เข้าถึงข้อมูลนี้")
        if response.status_code == 404:
            raise BookingAPIError("ไม่พบข้อมูลที่ต้องการ")
        if response.status_code == 429:
            # มักแปลว่า X-Internal-Key ว่างหรือไม่ตรงกับ backend เลยโดนนับ rate limit
            raise BookingAPIError(
                "ระบบจองมีคำขอมากเกินไป กรุณาลองใหม่อีกครั้งในสักครู่"
            )
        if response.status_code >= 400:
            raise BookingAPIError(f"ระบบจองตอบกลับด้วยสถานะ {response.status_code}")

        return response.json()

    # --- endpoint สาธารณะ (ไม่ต้องใช้ token) ---

    async def search_rooms(self, check_in: str, check_out: str) -> list[dict[str, Any]]:
        """ค้นหาห้องที่ว่างจริงในช่วง [check_in, check_out)

        **ต้องส่งวันที่ทั้งคู่เสมอ** — ถ้าเรียกโดยไม่ใส่วันที่ API จะข้ามการเช็ค
        ว่าชนกับ booking ที่มีอยู่หรือไม่ แล้วคืนทุกห้องที่ไม่ได้ปิดซ่อม
        ซึ่งแปลว่า "ไม่ได้ปิดซ่อม" ไม่ใช่ "ว่างให้จอง" คนละความหมายกันคนละเรื่อง

        **ไม่ส่ง guests ต่อเข้า API โดยตั้งใจ** — API กรองห้องที่ capacity < guests ทิ้ง
        ลูกค้ามา 3 คนจึงได้ผลว่างเปล่า (ยืนยันกับ API จริงแล้ว) ทั้งที่ยังมีทางเลือก
        คือจองห้อง capacity 2 แล้วโทรขอเตียงเสริม หรือจอง 2 ห้อง
        ต้องได้ห้องทั้งหมดมาก่อน แล้วให้โมเดลเทียบกับ capacity ตามกติกาใน SYSTEM_PROMPT
        """
        rooms = await self._get(
            "/rooms", params={"checkIn": check_in, "checkOut": check_out}
        )
        return [_trim_room(room) for room in rooms]

    async def quote_stay(
        self,
        check_in: str,
        check_out: str,
        rooms: int = 1,
        room_name: str | None = None,
    ) -> dict[str, Any]:
        """คิดยอดค่าห้องของการเข้าพักหนึ่งครั้ง โดยดึงราคาห้องสดจากระบบจอง

        **ทำไมต้องมีฟังก์ชันนี้ แทนที่จะให้โมเดลคูณเลขเอง**
        ทดสอบจริงแล้วโมเดลคิดเลขผิดเป็นระยะ เช่นตอบว่า "650 x 1 ห้อง x 2 คืน = 2,600 บาท"
        (ที่ถูกคือ 1,300) ผิดแบบนี้ลูกค้าไม่มีทางจับได้เพราะสูตรในวงเล็บเขียนถูก
        และมันยังนับจำนวนคืนจาก "มา 3 วัน" ไม่ตรงกันในแต่ละรอบด้วย
        พอย้ายการคูณและการนับคืนมาไว้ในโค้ด ผลลัพธ์เท่ากันทุกครั้งโดยนิยาม

        จำนวนคืนนับแบบ half-open [check_in, check_out) ให้ตรงกับที่ระบบจองใช้
        เข้าวันที่ 10 ออกวันที่ 12 = 2 คืน

        **ไม่คิดค่าเตียงเสริมโดยตั้งใจ** — ระบบจองไม่มีเรื่องเตียงเสริมเลย ยอดที่แอปเก็บ
        (totalPrice) คือค่าห้องอย่างเดียว ส่วนเตียงเสริมลูกค้าโทรขอแล้วจ่ายที่รีสอร์ทวันเช็คอิน
        ถ้ารวมค่าเตียงเสริมเข้ามาในยอดนี้ ตัวเลขที่บอทบอกจะไม่ตรงกับยอดที่ต้องโอนในแอป
        """
        try:
            start = date.fromisoformat(check_in)
            end = date.fromisoformat(check_out)
        except ValueError as exc:
            raise BookingAPIError(f"รูปแบบวันที่ไม่ถูกต้อง ต้องเป็น YYYY-MM-DD: {exc}") from exc

        nights = (end - start).days
        if nights < 1:
            raise BookingAPIError("วันออกต้องอยู่หลังวันเข้าพักอย่างน้อย 1 วัน")
        if rooms < 1:
            raise BookingAPIError("จำนวนห้องต้องเป็น 1 ห้องขึ้นไป")

        available = await self.search_rooms(check_in, check_out)
        if not available:
            raise BookingAPIError("ไม่มีห้องว่างในช่วงวันที่นี้")
        if len(available) < rooms:
            raise BookingAPIError(
                f"ช่วงวันที่นี้มีห้องว่างเพียง {len(available)} ห้อง ไม่พอสำหรับ {rooms} ห้อง"
            )

        if room_name:
            chosen = next((r for r in available if r["name"] == room_name), None)
            if chosen is None:
                raise BookingAPIError(f"ห้อง {room_name} ไม่ว่างในช่วงวันที่นี้")
        else:
            # ไม่ระบุห้องก็คิดจากห้องที่ถูกที่สุดที่ว่างอยู่ แล้วบอกไปด้วยว่าใช้ห้องไหนคิด
            chosen = min(available, key=lambda r: r["pricePerNight"])

        price_per_night = chosen["pricePerNight"]

        return {
            "checkIn": check_in,
            "checkOut": check_out,
            "nights": nights,
            "roomName": chosen["name"],
            "roomTypeLabel": chosen["typeLabel"],
            "pricePerNight": price_per_night,
            "rooms": rooms,
            "total": price_per_night * rooms * nights,
            "note": "ยอดนี้เป็นค่าห้องที่ชำระในแอป ไม่รวมเตียงเสริม",
        }

    async def list_rooms(self) -> list[dict[str, Any]]:
        """ทุกห้องที่ไม่ได้ปิดซ่อม โดยไม่สนว่าว่างหรือไม่ — ใช้หาว่าห้องไหน "มีแต่ไม่ว่าง"

        ห้ามใช้ตอบว่าห้องว่าง (เหตุผลอยู่ใน docstring ของ search_rooms)
        """
        rooms = await self._get("/rooms")
        return [_trim_room(room) for room in rooms]

    async def get_room(self, room_id: str) -> dict[str, Any]:
        """รายละเอียดห้องเดียว ได้แม้ห้องปิดซ่อม — ใช้เติมข้อมูลห้องให้การจองของลูกค้า"""
        return _trim_room(await self._get(f"/rooms/{room_id}"))

    async def list_restaurants(self) -> list[dict[str, Any]]:
        restaurants = await self._get("/restaurants")
        return [_trim_restaurant(item) for item in restaurants]

    async def get_payment_info(self) -> dict[str, Any]:
        """ข้อมูลการชำระเงิน รวม holdMinutes (ระยะเวลาที่ booking ค้างจ่ายถูกกันห้องไว้)

        holdMinutes ต้องอ่านสด ห้าม hardcode เพราะฝั่ง backend ปรับค่านี้ได้
        """
        return await self._get("/payment/info")

    # --- endpoint ที่ผูกกับตัวตนลูกค้า (ต้องมี token ของลูกค้าคนนั้น) ---

    async def get_my_bookings(self, token: str) -> list[dict[str, Any]]:
        bookings = await self._get("/bookings/me", token=token)
        return [_trim_booking(booking) for booking in bookings]

    async def get_booking_payment(self, booking_id: str, token: str) -> dict[str, Any]:
        """สถานะการชำระเงินของ booking หนึ่งรายการ

        booking.paymentStatus เป็นค่าหยาบ (unpaid/paid/refunded) ส่วนตัวที่ตอบคำถาม
        "เงินเข้าหรือยัง" ได้จริงคือ status ของ payment row ที่ละเอียดกว่า
        โดยเฉพาะ awaiting_verification ที่แปลว่าโอนแล้วแต่รอเจ้าหน้าที่ตรวจสลิป
        """
        return await self._get(f"/bookings/{booking_id}/payment", token=token)


#: แปลง type ที่ระบบจองใช้ ให้เป็นคำที่ลูกค้าพูดจริง
#:
#: ลูกค้าถามว่า "อยากได้เตียงเดี่ยว มีห้องว่างกี่ห้อง" แล้วบอกยกมาทั้ง 5 ห้อง
#: พร้อมบอกว่าเป็นเตียงเดี่ยวหมด ทั้งที่ผล tool มี type ติดมาครบทุกห้อง
#: สาเหตุคือ type เป็นคำอังกฤษ (single/twin) คนละคำกับที่ลูกค้าพิมพ์ โมเดลจึงไม่ได้
#: เอามาใช้กรอง การส่งคำไทยไปด้วยทำให้มันจับคู่กับคำถามได้ตรง ๆ
#: (หลักเดียวกับ quote_stay — คำนวณ/แปลงให้เสร็จในโค้ด อย่าฝากไว้กับ prompt)
#:
#: ถ้า backend เพิ่ม type ใหม่ที่ไม่มีในนี้ จะตกไปใช้ค่าดิบ ซึ่งยังอ่านออกแม้จะไม่สวย
ROOM_TYPE_LABELS = {
    "single": "เตียงเดี่ยว (มีเตียงใหญ่ 1 เตียง)",
    "twin": "เตียงคู่ (มีเตียงแยก 2 เตียง)",
}


def _trim_room(room: dict[str, Any]) -> dict[str, Any]:
    """เก็บเฉพาะ field ที่โมเดลต้องใช้ตอบ แล้วเติมค่าที่คำนวณได้ให้เสร็จ

    imageUrls/createdAt/updatedAt ไม่ช่วยตอบคำถามเลยแต่กิน token ทุกครั้งที่เรียก tool

    **ทำไมไม่มี maxGuestsWithExtraBed แล้ว** — เวอร์ชันก่อนส่ง capacity + 1 ไปให้โมเดล
    ผลคือบอทบอกลูกค้าว่า "จองห้องเดียว 3 คนได้" แต่ POST /api/bookings ตอบ 400 ทันที
    เมื่อ guests > capacity เพราะระบบจองไม่มีเรื่องเตียงเสริม ตอนนี้ capacity คือจำนวนสูงสุด
    ที่จองในแอปได้ ส่วนเตียงเสริมเป็นเรื่องที่ลูกค้าโทรขอเอง (กติกาอยู่ใน SYSTEM_PROMPT ข้อ 5)

    amenities ต้องส่งไปด้วยเสมอ เพราะแต่ละห้องไม่เหมือนกัน (บางห้องมีตู้เย็น บางห้องไม่มี)
    คลังความรู้จึงไม่มีรายการของในห้องแล้ว
    """
    return {
        "id": room["id"],
        "name": room["name"],
        "type": room["type"],
        "typeLabel": ROOM_TYPE_LABELS.get(room["type"], room["type"]),
        "pricePerNight": room["pricePerNight"],
        "capacity": room["capacity"],
        "description": room.get("description", ""),
        "amenities": room.get("amenities", []),
    }


def _trim_restaurant(restaurant: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": restaurant["name"],
        "cuisine": restaurant.get("cuisine", ""),
        "rating": restaurant.get("rating"),
        "priceRange": restaurant.get("priceRange", ""),
        "description": restaurant.get("description", ""),
        "address": restaurant.get("address", ""),
    }


def _trim_booking(booking: dict[str, Any]) -> dict[str, Any]:
    # customerName ถูกยกขึ้นไปไว้ระดับบนสุดใน group_bookings ไม่ได้ซ้ำทุกรายการ
    # roomId ใช้ดึงรายละเอียดห้อง (ชนิดเตียง ของในห้อง) ให้การจองที่ยังไม่ถึง
    return {
        "id": booking["id"],
        "customerName": booking.get("customerName", ""),
        "roomId": booking.get("roomId", ""),
        "roomName": booking.get("roomName", ""),
        "checkIn": booking["checkIn"],
        "checkOut": booking["checkOut"],
        "nights": booking.get("nights"),
        "guests": booking.get("guests"),
        "totalPrice": booking.get("totalPrice"),
        "status": booking.get("status"),
        "paymentStatus": booking.get("paymentStatus"),
    }


#: สถานะที่ยังถือว่า "จองอยู่" ถ้ายังไม่เลยวันเช็คเอาท์ — checked_in แยกไว้ต่างหาก
#: เพราะลูกค้ากำลังพักอยู่ นับเป็นปัจจุบันเสมอจนกว่าเจ้าหน้าที่จะกดเช็คเอาท์
_ACTIVE_BOOKING_STATUSES = frozenset({"pending", "approved"})


def group_bookings(
    bookings: list[dict[str, Any]], today: str, include_past: bool = False
) -> dict[str, Any]:
    """แยกการจองเป็น "ที่ยังไม่ถึง/กำลังพัก" กับ "ประวัติ" ให้เสร็จในโค้ด

    **ทำไมไม่ส่งทั้งหมดให้โมเดลกรองเอง** — ทดสอบจริง ลูกค้าถาม "ฉันจองอะไรไว้บ้าง"
    บอทไล่ทุกรายการรวมที่เช็คเอาท์ไปแล้วและที่เลยวันไปแล้ว 7 รายการ ทั้งที่ลูกค้า
    อยากรู้แค่ที่กำลังจะมาถึง โมเดลเทียบวันที่กับ "วันนี้" ไม่แม่น (หลักเดียวกับ quote_stay)

    ปัจจุบัน = checked_in หรือ (pending/approved และ checkOut > วันนี้)
    นอกนั้นเป็นประวัติ (เช็คเอาท์แล้ว ยกเลิกแล้ว หรือเลยวันเช็คเอาท์ไปแล้ว)
    วันที่เป็น YYYY-MM-DD จึงเทียบเป็น string ได้ตรง ๆ

    **stayedCount แยกจาก pastBookingsCount** — ทดสอบจริง ลูกค้าถาม "เคยพักกี่ครั้ง"
    บอทตอบ 14 ครั้งจาก pastBookingsCount ทั้งที่ 11 รายการในนั้นคือที่ยกเลิก
    เช็คเอาท์จริงแค่ 3 ครั้ง จึงนับแยกให้เสร็จในโค้ด

    **recentPastBookings ส่งไปเสมอ** — โมเดลแทบไม่เคยส่ง include_past มาเอง
    ลูกค้าถาม "การจองที่ยกเลิกไปได้เงินคืนหรือยัง" แล้วบอทตอบเป็นนโยบายลอย ๆ
    ทั้งที่มีรายการที่ refunded อยู่จริง 5 รายการล่าสุดพอตอบคำถามแบบนี้ได้โดยไม่กิน token มาก
    """
    current: list[dict[str, Any]] = []
    past: list[dict[str, Any]] = []
    customer_name = ""
    for booking in bookings:
        booking = dict(booking)
        customer_name = customer_name or booking.pop("customerName", "")
        booking.pop("customerName", None)
        is_current = booking.get("status") == "checked_in" or (
            booking.get("status") in _ACTIVE_BOOKING_STATUSES
            and str(booking.get("checkOut", ""))[:10] > today
        )
        (current if is_current else past).append(booking)

    current.sort(key=lambda b: str(b.get("checkIn", "")))
    past.sort(key=lambda b: str(b.get("checkIn", "")), reverse=True)
    result: dict[str, Any] = {
        "today": today,
        "customerName": customer_name,
        "currentBookings": current,
        "pastBookingsCount": len(past),
        "stayedCount": sum(1 for b in past if b.get("status") == "checked_out"),
        "cancelledCount": sum(1 for b in past if b.get("status") == "cancelled"),
        "recentPastBookings": past[:_RECENT_PAST_LIMIT],
    }
    if include_past:
        result["pastBookings"] = past
    return result


_RECENT_PAST_LIMIT = 5

_THAI_MONTHS = (
    "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
)
_EN_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
_BUDDHIST_ERA_OFFSET = 543


def format_date(value: date, language: str = "th") -> str:
    if language == "th":
        return f"{value.day} {_THAI_MONTHS[value.month - 1]} {value.year + _BUDDHIST_ERA_OFFSET}"
    return f"{value.day} {_EN_MONTHS[value.month - 1]} {value.year}"


def format_stay_dates(check_in: str, check_out: str, language: str = "th") -> str:
    """ข้อความวันเข้า-ออกสำเร็จรูปให้โมเดลพูดตาม

    ทดสอบจริงบอทเขียนวันที่ปนกันสามแบบในบทสนทนาเดียว (2569, 2027, 2026-10-07)
    และถาม "13-15 เมษา" แล้วตอบกลับว่า "13-16 เมษายน" การแปลงวันที่ให้เสร็จในโค้ด
    ทำให้ลูกค้าเห็นวันที่ที่ใช้ค้นจริงในรูปแบบเดียวกันทุกครั้ง
    """
    start, end = date.fromisoformat(check_in), date.fromisoformat(check_out)
    if language == "th":
        return f"เข้าพัก {format_date(start)} ออก {format_date(end)}"
    return f"check-in {format_date(start, 'en')}, check-out {format_date(end, 'en')}"


def plan_rooms_for_guests(
    available: list[dict[str, Any]],
    guests: int,
    nights: int,
    extra_bed_rate: int | None,
    max_extra_beds_per_room: int = 1,
) -> dict[str, Any]:
    """หาว่าลูกค้า N คนต้องจองกี่ห้อง ทั้งแบบไม่ขอเตียงเสริมและแบบขอเตียงเสริม

    **ทำไมต้องคิดในโค้ด** — ทดสอบจริง 3 รอบ โมเดลคิดเรื่องนี้ผิดซ้ำทุกครั้งแม้กติกา
    ใน SYSTEM_PROMPT ข้อ 5 จะเขียนไว้ละเอียด:
    - "มา 4 คน ห้องเดียวแล้วขอเตียงเสริมได้ไหม" -> "ได้ค่ะ" (ห้องเดียวรับได้แค่ 3)
    - "มา 3 คน รวมเตียงเสริมจ่ายเท่าไหร่" -> คิดเป็น 2 ห้อง 2,600 แทน 1 ห้อง + เตียงเสริม
    - "มา 6 คน" -> แนะนำ 3 ห้อง ทั้งที่ 2 ห้อง + เตียงเสริม 2 เตียงถูกกว่า

    เลือกห้องจากถูกไปแพง แล้วหาจำนวนห้องน้อยที่สุดที่พอสำหรับแต่ละแบบ
    ค่าเตียงเสริมรับเข้ามาเป็นพารามิเตอร์ เพราะคลังความรู้เป็นเจ้าของตัวเลขนั้น
    ส่วน "ห้องละไม่เกิน 1 เตียง" มาจากคลังความรู้เช่นกัน (booking_policy.md)
    """
    rooms = sorted(available, key=lambda r: (r["pricePerNight"], -r["capacity"]))
    max_capacity = max((r["capacity"] for r in rooms), default=0)

    def cheapest_prefix(extra_beds_allowed: bool) -> dict[str, Any] | None:
        total_capacity = 0
        for count, room in enumerate(rooms, start=1):
            total_capacity += room["capacity"]
            reach = total_capacity + (count * max_extra_beds_per_room if extra_beds_allowed else 0)
            if reach >= guests:
                chosen = rooms[:count]
                extra_beds = max(0, guests - total_capacity) if extra_beds_allowed else 0
                room_total = sum(r["pricePerNight"] for r in chosen) * nights
                if not extra_beds:
                    bed_total: int | None = 0
                elif extra_bed_rate is None:
                    bed_total = None  # ไม่รู้ค่าเตียงเสริม ห้ามเดา ให้โมเดลบอกว่าไม่มีข้อมูล
                else:
                    bed_total = extra_beds * extra_bed_rate * nights
                return {
                    "rooms": count,
                    "roomNames": [r["name"] for r in chosen],
                    "guestsToEnterInApp": min(guests, total_capacity),
                    "extraBeds": extra_beds,
                    "payInApp": room_total,
                    "payAtResortForExtraBeds": bed_total,
                    "grandTotal": None if bed_total is None else room_total + bed_total,
                }
        return None

    without_beds = cheapest_prefix(extra_beds_allowed=False)
    with_beds = cheapest_prefix(extra_beds_allowed=True)
    options = [opt for opt in (without_beds, with_beds) if opt is not None]
    # แบบขอเตียงเสริมที่สุดท้ายไม่ต้องใช้เตียงเสริมเลย ก็คือแบบแรกซ้ำ
    if len(options) == 2 and options[1]["extraBeds"] == 0:
        options = options[:1]

    return {
        "guests": guests,
        "nights": nights,
        "maxGuestsPerRoomInApp": max_capacity,
        "maxGuestsPerRoomWithExtraBed": max_capacity + max_extra_beds_per_room,
        "fitsInOneRoom": guests <= max_capacity,
        "fitsInOneRoomWithExtraBed": guests <= max_capacity + max_extra_beds_per_room,
        "enoughRoomsAvailable": bool(options),
        "availableRoomCount": len(rooms),
        "extraBedRatePerNight": extra_bed_rate,
        "options": options,
    }


@lru_cache
def get_booking_api_client() -> BookingAPIClient:
    return BookingAPIClient()
