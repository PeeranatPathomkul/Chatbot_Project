"""
เฉลยของข้อมูลสด — ดึงจาก Booking API ตอนรัน แล้วบันทึกเป็น snapshot ลง raw

**ทำไมดึงสองครั้ง (ก่อนและหลังถามบอท)** ห้องว่างเปลี่ยนได้ระหว่างที่บอทกำลังตอบ (มีคนจองพอดี)
ถ้าใช้ snapshot ครั้งเดียวแล้วผลไม่ตรง จะแยกไม่ออกว่าบอทผิดหรือข้อมูลเปลี่ยน จึงเทียบ fingerprint
ของสองครั้ง ถ้าต่างกันข้อนั้นเป็น indeterminate และไม่ถูกนับใน key-fact accuracy

กติกาเตียงเสริม (อัตราต่อคืน, สูงสุดกี่เตียงต่อห้อง) อ่านจาก booking_policy.md ของคลังความรู้ตอนรัน
ไม่ได้เขียนตัวเลขไว้ในโค้ด — ถ้าเจ้าของแก้ราคาในเอกสาร เฉลยก็เปลี่ยนตาม
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from functools import lru_cache
from typing import Any

from harness import dates
from harness.client import BookingAPIError, BookingClient
from harness.config import KNOWLEDGE_DIR

_RESTAURANT_CHECKS = {"restaurant_names_min", "restaurant_match"}
_PAYMENT_CHECKS = {"promptpay_id", "hold_minutes"}
#: สถานะการจองที่ยังถือว่า "จองอยู่" — นิยามเดียวกับระบบจอง (ดูหัวข้อสถานะการจองใน README)
_ACTIVE_STATUSES = {"pending", "approved"}


@lru_cache
def kb_extra_bed_rules() -> dict[str, int]:
    text = (KNOWLEDGE_DIR / "th" / "booking_policy.md").read_text(encoding="utf-8")
    rate = re.search(r"คืนละ\s*([\d,]+)\s*บาทต่อเตียง", text)
    per_room = re.search(r"ห้องละไม่เกิน\s*(\d+)\s*เตียง", text)
    if not rate or not per_room:
        raise RuntimeError("อ่านกติกาเตียงเสริมจาก th/booking_policy.md ไม่ได้")
    return {"rate_per_night": int(rate.group(1).replace(",", "")), "max_per_room": int(per_room.group(1))}


def is_current(booking: dict[str, Any], today: str) -> bool:
    """การจองที่ยังไม่ถึงหรือกำลังพัก: checked_in หรือ pending/approved ที่ checkOut > วันนี้"""
    status = booking.get("status")
    return status == "checked_in" or (status in _ACTIVE_STATUSES and str(booking.get("checkOut", ""))[:10] > today)


def _check_names(item: dict[str, Any]) -> set[str]:
    names = set()
    for check in (item.get("live") or {}).get("checks", []):
        names.add(next(iter(check)) if isinstance(check, dict) else check)
    return names


class GroundTruth:
    """ดึงและ cache เฉลยของหนึ่งรอบ — ข้อมูลที่ไม่เปลี่ยน (รายชื่อห้อง ร้านอาหาร พร้อมเพย์) ดึงครั้งเดียว"""

    def __init__(self, booking: BookingClient, today: date):
        self.booking = booking
        self.today = today
        self._static: dict[str, Any] = {}
        self._other: dict[str, Any] | None = None

    async def _cached(self, key: str, fetch) -> Any:
        if key not in self._static:
            self._static[key] = await fetch()
        return self._static[key]

    async def customer(self, who: str) -> dict[str, Any]:
        """การจองทั้งหมดของบัญชีทดสอบ + รายละเอียดของการจองปัจจุบันที่ใกล้ที่สุด"""
        today = self.today.isoformat()
        raw = await self.booking.my_bookings(who)
        bookings = [
            {k: b.get(k) for k in ("id", "customerName", "roomId", "roomName", "checkIn", "checkOut",
                                   "nights", "guests", "totalPrice", "status", "paymentStatus")}
            for b in raw
        ]
        current = sorted((b for b in bookings if is_current(b, today)), key=lambda b: str(b["checkIn"]))
        nearest = current[0] if current else None
        snapshot: dict[str, Any] = {
            "customerName": next((b["customerName"] for b in bookings if b.get("customerName")), ""),
            "bookings": bookings,
            "current_ids": [b["id"] for b in current],
            "nearest_id": nearest["id"] if nearest else None,
            "stayed_count": sum(1 for b in bookings if b.get("status") == "checked_out"),
        }
        if nearest:
            snapshot["nearest_payment"] = await self.booking.booking_payment(nearest["id"], who)
            try:
                snapshot["nearest_room"] = await self.booking.room(nearest["roomId"]) if nearest.get("roomId") else None
            except BookingAPIError:
                snapshot["nearest_room"] = None
        return snapshot

    async def other_customer(self) -> dict[str, Any]:
        if self._other is None:
            self._other = await self.customer("other")
        return self._other

    async def other_booking_id(self) -> str | None:
        other = await self.other_customer()
        ids = [b["id"] for b in other["bookings"]]
        return (other["current_ids"] or ids or [None])[0]

    async def snapshot(self, item: dict[str, Any]) -> dict[str, Any]:
        snap: dict[str, Any] = {}
        live = item.get("live") or {}
        checks = _check_names(item)
        if live.get("dates"):
            resolved = dates.resolve_args(live["dates"], self.today)
            snap["dates"] = resolved
            snap["available_rooms"] = await self.booking.available_rooms(resolved["check_in"], resolved["check_out"])
        if live or item.get("auth_checks") or item["suite"] in ("live", "multi_source", "multi_turn", "auth"):
            snap["all_rooms"] = await self._cached("all_rooms", self.booking.all_rooms)
        if checks & _RESTAURANT_CHECKS:
            snap["restaurants"] = await self._cached("restaurants", self.booking.restaurants)
        if checks & _PAYMENT_CHECKS:
            snap["payment_info"] = await self._cached("payment_info", self.booking.payment_info)
        if checks & {"guests_plan", "grand_total", "extra_bed_total"}:
            snap["kb_extra_bed"] = kb_extra_bed_rules()
        security = set(item.get("checks") or [])
        if item.get("auth") == "customer" or "no_customer_data" in security:
            snap["customer"] = await self.customer("customer")
        if "no_other_customer_data" in security:
            snap["other"] = await self.other_customer()
        return snap


def fingerprint(snapshot: dict[str, Any]) -> str:
    """ส่วนของเฉลยที่เปลี่ยนได้ระหว่างถาม — ใช้เทียบ snapshot ก่อน/หลัง"""
    dynamic = {
        "available": sorted((r["name"], r["pricePerNight"]) for r in snapshot.get("available_rooms", [])),
        "customer": sorted(
            (b["id"], b.get("status"), b.get("paymentStatus"))
            for b in (snapshot.get("customer") or {}).get("bookings", [])
        ),
        "payment": json.dumps((snapshot.get("customer") or {}).get("nearest_payment"), sort_keys=True, default=str),
    }
    return hashlib.sha256(json.dumps(dynamic, sort_keys=True, default=str).encode()).hexdigest()[:16]
