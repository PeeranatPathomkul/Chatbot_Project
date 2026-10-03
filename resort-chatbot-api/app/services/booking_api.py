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
    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        self.base_url = (base_url or settings.booking_api_base_url).rstrip("/")
        self.timeout = timeout or settings.booking_api_timeout_seconds

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> Any:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
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
        if response.status_code >= 400:
            raise BookingAPIError(f"ระบบจองตอบกลับด้วยสถานะ {response.status_code}")

        return response.json()

    # --- endpoint สาธารณะ (ไม่ต้องใช้ token) ---

    async def search_rooms(self, check_in: str, check_out: str) -> list[dict[str, Any]]:
        """ค้นหาห้องที่ว่างจริงในช่วง [check_in, check_out)

        **ต้องส่งวันที่ทั้งคู่เสมอ** — ถ้าเรียกโดยไม่ใส่วันที่ API จะข้ามการเช็ค
        ว่าชนกับ booking ที่มีอยู่หรือไม่ แล้วคืนทุกห้องที่ไม่ได้ปิดซ่อม
        ซึ่งแปลว่า "ไม่ได้ปิดซ่อม" ไม่ใช่ "ว่างให้จอง" คนละความหมายกันคนละเรื่อง

        **ไม่ส่ง guests ต่อเข้า API โดยตั้งใจ** — capacity ใน DB คือจำนวนคนที่นอนได้
        ด้วยเตียงที่มีอยู่ในห้อง (2) แต่ห้องรับได้จริงถึง 3 ท่านโดยขอเตียงเสริม
        การส่ง ?guests=3 จึงกรองห้องจริงทิ้งทั้งหมด (ยืนยันกับ API จริงแล้ว)
        ปล่อยให้โมเดลเอา capacity ไปคิดเรื่องเตียงเสริมเองตามกติกาใน SYSTEM_PROMPT
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
        extra_beds: int = 0,
        extra_bed_price_per_night: float | None = None,
        room_name: str | None = None,
    ) -> dict[str, Any]:
        """คิดยอดรวมของการเข้าพักหนึ่งครั้ง โดยดึงราคาห้องสดจากระบบจอง

        **ทำไมต้องมีฟังก์ชันนี้ แทนที่จะให้โมเดลคูณเลขเอง**
        ทดสอบจริงแล้วโมเดลคิดเลขผิดเป็นระยะ เช่นตอบว่า "650 x 1 ห้อง x 2 คืน = 2,600 บาท"
        (ที่ถูกคือ 1,300) ผิดแบบนี้ลูกค้าไม่มีทางจับได้เพราะสูตรในวงเล็บเขียนถูก
        และมันยังนับจำนวนคืนจาก "มา 3 วัน" ไม่ตรงกันในแต่ละรอบด้วย
        พอย้ายการคูณและการนับคืนมาไว้ในโค้ด ผลลัพธ์เท่ากันทุกครั้งโดยนิยาม

        จำนวนคืนนับแบบ half-open [check_in, check_out) ให้ตรงกับที่ระบบจองใช้
        เข้าวันที่ 10 ออกวันที่ 12 = 2 คืน
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
        if extra_beds < 0:
            raise BookingAPIError("จำนวนเตียงเสริมติดลบไม่ได้")
        if extra_beds > rooms * EXTRA_BEDS_PER_ROOM:
            raise BookingAPIError(
                f"ขอเตียงเสริมได้ห้องละ {EXTRA_BEDS_PER_ROOM} เตียง "
                f"จอง {rooms} ห้องจึงขอได้สูงสุด {rooms * EXTRA_BEDS_PER_ROOM} เตียง"
            )
        if extra_beds > 0 and not extra_bed_price_per_night:
            # บังคับให้ไปเอาอัตราจริงมาก่อน ไม่งั้นโมเดลจะเดาตัวเลขเอง
            raise BookingAPIError(
                "ต้องระบุ extra_bed_price_per_night เมื่อขอเตียงเสริม "
                "ให้ค้นอัตราค่าเตียงเสริมจาก search_knowledge_base ก่อน"
            )

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
        room_total = price_per_night * rooms * nights
        extra_bed_total = (extra_bed_price_per_night or 0) * extra_beds * nights

        return {
            "checkIn": check_in,
            "checkOut": check_out,
            "nights": nights,
            "roomName": chosen["name"],
            "roomTypeLabel": chosen["typeLabel"],
            "pricePerNight": price_per_night,
            "rooms": rooms,
            "roomTotal": room_total,
            "extraBeds": extra_beds,
            "extraBedPricePerNight": extra_bed_price_per_night or 0,
            "extraBedTotal": extra_bed_total,
            "total": room_total + extra_bed_total,
            "note": (
                "ยอดนี้เป็นค่าห้องอย่างเดียว ยังไม่รวมเตียงเสริม"
                if extra_beds == 0
                else f"รวมค่าเตียงเสริม {extra_beds} เตียง x {nights} คืนแล้ว"
            ),
        }

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


#: จำนวนเตียงเสริมที่ขอได้ต่อห้อง — นโยบายของรีสอร์ท (ที่มา: คลังความรู้ room_types.md)
#: ค่าเตียงเสริมเป็นตัวเลขที่คลังความรู้เป็นเจ้าของ แต่ "ขอได้กี่เตียงต่อห้อง"
#: ต้องมาอยู่ตรงนี้ด้วย เพราะต้องใช้คำนวณ maxGuestsWithExtraBed ให้โมเดล
#: ถ้ารีสอร์ทเปลี่ยนนโยบาย ต้องแก้ทั้งสองที่ให้ตรงกัน
EXTRA_BEDS_PER_ROOM = 1

#: แปลง type ที่ระบบจองใช้ ให้เป็นคำที่ลูกค้าพูดจริง
#:
#: ลูกค้าถามว่า "อยากได้เตียงเดี่ยว มีห้องว่างกี่ห้อง" แล้วบอกยกมาทั้ง 5 ห้อง
#: พร้อมบอกว่าเป็นเตียงเดี่ยวหมด ทั้งที่ผล tool มี type ติดมาครบทุกห้อง
#: สาเหตุคือ type เป็นคำอังกฤษ (single/twin) คนละคำกับที่ลูกค้าพิมพ์ โมเดลจึงไม่ได้
#: เอามาใช้กรอง การส่งคำไทยไปด้วยทำให้มันจับคู่กับคำถามได้ตรง ๆ
#: (เหตุผลเดียวกับ maxGuestsWithExtraBed — คำนวณ/แปลงให้เสร็จในโค้ด อย่าฝากไว้กับ prompt)
#:
#: ถ้า backend เพิ่ม type ใหม่ที่ไม่มีในนี้ จะตกไปใช้ค่าดิบ ซึ่งยังอ่านออกแม้จะไม่สวย
ROOM_TYPE_LABELS = {
    "single": "เตียงเดี่ยว (มีเตียงใหญ่ 1 เตียง)",
    "twin": "เตียงคู่ (มีเตียงแยก 2 เตียง)",
}


def _trim_room(room: dict[str, Any]) -> dict[str, Any]:
    """เก็บเฉพาะ field ที่โมเดลต้องใช้ตอบ แล้วเติมค่าที่คำนวณได้ให้เสร็จ

    imageUrls/createdAt/updatedAt ไม่ช่วยตอบคำถามเลยแต่กิน token ทุกครั้งที่เรียก tool

    **ทำไมต้องคำนวณ maxGuestsWithExtraBed ให้ ไม่ปล่อยให้โมเดลบวกเอง**
    ตอนแรกบอกกติกาไว้ใน system prompt ว่า "จำนวนสูงสุดต่อห้อง = capacity + 1"
    แต่ทดสอบจริงแล้วโมเดลทำตามบ้างไม่ทำตามบ้าง คำถามเดียวกัน ("มา 3 คน พักห้องเดียว
    ได้ไหม") รอบหนึ่งตอบว่าได้พร้อมคิดค่าเตียงเสริมถูกต้อง อีกรอบตอบว่าไม่ได้
    เพราะมันยึดเลข capacity: 2 ที่เห็นเป็นตัวเลขชัด ๆ ใน JSON มากกว่ากติกาที่เป็นร้อยแก้ว
    การส่งค่าที่คำนวณเสร็จแล้วไปเลยตัดปัญหานี้ทิ้งทั้งก้อน
    """
    capacity = room["capacity"]
    return {
        "id": room["id"],
        "name": room["name"],
        "type": room["type"],
        "typeLabel": ROOM_TYPE_LABELS.get(room["type"], room["type"]),
        "pricePerNight": room["pricePerNight"],
        "capacity": capacity,
        "maxGuestsWithExtraBed": capacity + EXTRA_BEDS_PER_ROOM,
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
    return {
        "id": booking["id"],
        "roomName": booking.get("roomName", ""),
        "checkIn": booking["checkIn"],
        "checkOut": booking["checkOut"],
        "nights": booking.get("nights"),
        "guests": booking.get("guests"),
        "totalPrice": booking.get("totalPrice"),
        "status": booking.get("status"),
        "paymentStatus": booking.get("paymentStatus"),
    }


@lru_cache
def get_booking_api_client() -> BookingAPIClient:
    return BookingAPIClient()
