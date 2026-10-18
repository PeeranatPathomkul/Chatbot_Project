"""
HTTP client ของ harness: ยิงแชทบอท (FastAPI ตรง ไม่ผ่าน NestJS) และ Booking API (ใช้เป็นเฉลย)

**retry** — แอปไม่ได้จับ error ของ LLM: เมื่อ Typhoon ตอบ 429 แอปจะคืน 500 ให้เรา
ตอนวัดความแม่นยำจึง retry ทั้ง 429 และ 5xx แบบ exponential backoff แล้วบันทึกจำนวนครั้งที่ลองไว้ใน raw
(ตัวเลข error ยังเห็นได้) ส่วน load test ไม่ retry เพื่อวัด error rate จริง

**token ลูกค้า** — access token อายุ 15 นาที ล็อกอินใหม่เองเมื่อใกล้หมดอายุหรือเจอ 401
token ไม่ถูกเขียนลงไฟล์ (ลงทะเบียนไว้กับ config.redact)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from harness.config import env, register_secret, settings

#: ล็อกอินใหม่ก่อนหมดอายุ (token อายุ 15 นาที)
_TOKEN_MAX_AGE_SECONDS = 12 * 60


@dataclass
class ChatResult:
    status_code: int | None
    latency_ms: float
    body: dict[str, Any] | None
    trace: dict[str, Any] | None
    attempts: int
    error: str | None = None
    timed_out: bool = False
    request_ts: float = 0.0
    attempt_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "latency_ms": round(self.latency_ms, 1),
            "attempts": self.attempts,
            "attempt_errors": self.attempt_errors,
            "error": self.error,
            "timed_out": self.timed_out,
            "request_ts": self.request_ts,
            "response": self.body,
            "trace": self.trace,
        }


class ChatClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.cfg = settings()

    async def ask(
        self,
        session_id: str,
        message: str,
        language: str,
        token: str | None = None,
        retry: bool = True,
    ) -> ChatResult:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = {"session_id": session_id, "message": message, "language": language}
        attempt_errors: list[str] = []
        max_attempts = self.cfg.max_retries + 1 if retry else 1
        result: ChatResult | None = None
        for attempt in range(1, max_attempts + 1):
            started_ts = time.time()
            started = time.perf_counter()
            try:
                response = await self.client.post(
                    self.cfg.query_url, json=payload, headers=headers, timeout=self.cfg.timeout_seconds
                )
                latency = (time.perf_counter() - started) * 1000
            except httpx.TimeoutException:
                latency = (time.perf_counter() - started) * 1000
                result = ChatResult(None, latency, None, None, attempt, "timeout", True, started_ts)
                attempt_errors.append("timeout")
            except httpx.HTTPError as exc:
                latency = (time.perf_counter() - started) * 1000
                result = ChatResult(None, latency, None, None, attempt, f"{type(exc).__name__}", False, started_ts)
                attempt_errors.append(type(exc).__name__)
            else:
                body = None
                try:
                    body = response.json()
                except ValueError:
                    pass
                trace = await self._trace(response.headers.get("X-Eval-Trace-Id"))
                error = None if response.status_code == 200 else f"http_{response.status_code}"
                result = ChatResult(response.status_code, latency, body, trace, attempt, error, False, started_ts)
                if response.status_code == 200:
                    break
                attempt_errors.append(error)
                if response.status_code not in (429, 500, 502, 503, 504):
                    break
            if attempt < max_attempts:
                await asyncio.sleep(min(60.0, 2.0 * 2 ** (attempt - 1)))
        assert result is not None
        result.attempt_errors = attempt_errors
        return result

    async def _trace(self, trace_id: str | None) -> dict[str, Any] | None:
        if not trace_id:
            return None
        try:
            response = await self.client.get(f"{self.cfg.api_base_url}/_eval/trace/{trace_id}", timeout=10)
            return response.json()
        except (httpx.HTTPError, ValueError):
            return None

    async def info(self) -> dict[str, Any] | None:
        try:
            response = await self.client.get(f"{self.cfg.api_base_url}/_eval/info", timeout=10)
        except httpx.HTTPError:
            return None
        return response.json() if response.status_code == 200 else None

    async def health(self) -> int | None:
        try:
            response = await self.client.get(f"{self.cfg.api_base_url}/api/v1/chatbot/health", timeout=10)
            return response.status_code
        except httpx.HTTPError:
            return None


class BookingAPIError(RuntimeError):
    pass


class BookingClient:
    """เรียก Booking API ตรงเพื่อเอาเฉลย (GET ทั้งหมด ยกเว้น login)"""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.cfg = settings()
        self._tokens: dict[str, tuple[str, float]] = {}

    def _headers(self, token: str | None = None) -> dict[str, str]:
        headers = {}
        if self.cfg.booking_internal_key:
            headers["X-Internal-Key"] = self.cfg.booking_internal_key
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def get(self, path: str, params: dict[str, Any] | None = None, who: str | None = None) -> Any:
        for attempt in range(3):
            token = await self.token(who) if who else None
            try:
                response = await self.client.get(
                    f"{self.cfg.booking_base_url}{path}", params=params, headers=self._headers(token), timeout=15
                )
            except httpx.HTTPError as exc:
                raise BookingAPIError(f"เรียก Booking API ไม่ได้: {type(exc).__name__}") from exc
            if response.status_code == 401 and who and attempt == 0:
                self._tokens.pop(who, None)  # token หมดอายุ ล็อกอินใหม่แล้วลองอีกครั้ง
                continue
            if response.status_code == 429:
                await asyncio.sleep(5 * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise BookingAPIError(f"GET {path} -> {response.status_code}")
            return response.json()
        raise BookingAPIError(f"GET {path} ไม่สำเร็จหลังลองซ้ำ")

    async def token(self, who: str) -> str:
        """who = 'customer' หรือ 'other' (บัญชีจาก EVAL_<WHO>_EMAIL / PASSWORD)"""
        cached = self._tokens.get(who)
        if cached and time.time() - cached[1] < _TOKEN_MAX_AGE_SECONDS:
            return cached[0]
        prefix = f"EVAL_{who.upper()}"
        email, password = env(f"{prefix}_EMAIL"), env(f"{prefix}_PASSWORD")
        if not (email and password):
            raise BookingAPIError(f"ไม่มี {prefix}_EMAIL / {prefix}_PASSWORD")
        try:
            response = await self.client.post(
                f"{self.cfg.booking_base_url}/auth/login", json={"email": email, "password": password}, timeout=15
            )
        except httpx.HTTPError as exc:
            raise BookingAPIError(f"ล็อกอินไม่ได้: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise BookingAPIError(f"ล็อกอิน {who} ไม่สำเร็จ ({response.status_code})")
        token = response.json().get("accessToken")
        if not token:
            raise BookingAPIError("response ของ /auth/login ไม่มี accessToken")
        register_secret(token)
        self._tokens[who] = (token, time.time())
        return token

    # --- endpoint ที่ใช้เป็นเฉลย ---

    async def available_rooms(self, check_in: str, check_out: str) -> list[dict[str, Any]]:
        return [_trim_room(r) for r in await self.get("/rooms", {"checkIn": check_in, "checkOut": check_out})]

    async def all_rooms(self) -> list[dict[str, Any]]:
        return [_trim_room(r) for r in await self.get("/rooms")]

    async def room(self, room_id: str) -> dict[str, Any]:
        return _trim_room(await self.get(f"/rooms/{room_id}"))

    async def restaurants(self) -> list[dict[str, Any]]:
        return [
            {k: r.get(k) for k in ("name", "cuisine", "description", "priceRange")}
            for r in await self.get("/restaurants")
        ]

    async def payment_info(self) -> dict[str, Any]:
        info = await self.get("/payment/info")
        return {k: info.get(k) for k in ("promptPayId", "accountName", "holdMinutes")}

    async def my_bookings(self, who: str) -> list[dict[str, Any]]:
        return await self.get("/bookings/me", who=who)

    async def booking_payment(self, booking_id: str, who: str) -> Any:
        try:
            return await self.get(f"/bookings/{booking_id}/payment", who=who)
        except BookingAPIError as exc:
            return {"error": str(exc)}


def _trim_room(room: dict[str, Any]) -> dict[str, Any]:
    return {k: room.get(k) for k in ("id", "name", "type", "pricePerNight", "capacity", "amenities", "status")}
