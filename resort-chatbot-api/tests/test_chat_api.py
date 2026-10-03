"""
Test สำหรับ endpoint /api/v1/chatbot/query และ /api/v1/chatbot/health

หมายเหตุ: mock LLMClient, BookingAPIClient, VectorStore และ EmbeddingService ไว้ทั้งหมด
เพื่อไม่ให้ยิง API จริงหรือโหลดโมเดล embedding จริงตอนรัน test (ประหยัดเวลา/เงิน)

**FakeLLMClient คืนผลลัพธ์เป็นคิว** เพราะการตอบหนึ่งคำถามไม่ใช่การเรียก LLM ครั้งเดียว
อีกต่อไป แต่เป็นลูป: ขอเรียก tool -> รับผล -> ตอบ การ mock ด้วยค่าคืนค่าเดียว
จะทำให้ลูปวนไม่รู้จบเพราะโมเดลจำลองขอเรียก tool ซ้ำทุกครั้ง
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.prompts import NO_CONTEXT_ANSWER
from app.main import app
from app.services.llm_client import LLMResult, ToolCall
from app.services.rag_pipeline import RAGPipeline, get_rag_pipeline

#: chunk ตัวอย่างในรูปแบบที่ indexing-pipeline สร้างจริง — มีบรรทัดบริบทนำหน้า
SAMPLE_CHUNK_TEXT = (
    "[หมวด: นโยบาย | หัวข้อ: เวลาเช็คอินและเช็คเอาท์]\n"
    "เวลาเช็คอินคือ 14.00 น. เป็นต้นไป\n"
    "รับเช็คอินได้ถึงเวลา 22.00 น."
)

SAMPLE_ROOMS = [
    {
        "id": "a7e4e205-91d0-40cc-a245-fdf7f3047b8d",
        "name": "P1",
        "type": "single",
        "pricePerNight": 650,
        "capacity": 2,
        "description": "Single bedroom with a king-size bed.",
        "amenities": ["Wi-Fi", "Air conditioning"],
    }
]


class FakeLLMClient:
    """LLMClient จำลองที่คืนผลตามคิวที่กำหนดไว้ล่วงหน้า และบันทึก messages ทุกรอบ"""

    def __init__(self, results: list[LLMResult]):
        self.results = list(results)
        self.calls: list[dict] = []

    async def generate(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.results:
            return LLMResult(text="จบการสนทนาค่ะ", usage=None)
        return self.results.pop(0)


def usage(total: int = 150) -> dict[str, int]:
    return {"prompt_tokens": 120, "completion_tokens": total - 120, "total_tokens": total}


def build_pipeline(llm_results: list[LLMResult], hits: list[dict] | None = None):
    """สร้าง RAGPipeline ที่ทุก dependency ถูก mock ไว้"""
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = (
        hits
        if hits is not None
        else [
            {
                "doc_id": "th-booking_policy::0000",
                "text": SAMPLE_CHUNK_TEXT,
                "metadata": {"source": "th/booking_policy.md", "language": "th"},
                "distance": 0.12,
            }
        ]
    )

    mock_booking_client = AsyncMock()
    mock_booking_client.search_rooms.return_value = SAMPLE_ROOMS

    fake_llm = FakeLLMClient(llm_results)

    pipeline = RAGPipeline(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_client=fake_llm,
        booking_client=mock_booking_client,
    )
    return pipeline, fake_llm, mock_booking_client, mock_vector_store


def build_client(llm_results: list[LLMResult], hits: list[dict] | None = None) -> TestClient:
    pipeline, fake_llm, mock_booking_client, mock_vector_store = build_pipeline(
        llm_results, hits
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline

    client = TestClient(app)
    client.fake_llm = fake_llm  # type: ignore[attr-defined]
    client.mock_booking_client = mock_booking_client  # type: ignore[attr-defined]
    client.mock_vector_store = mock_vector_store  # type: ignore[attr-defined]
    return client


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def knowledge_flow(answer: str) -> list[LLMResult]:
    """โมเดลค้นคลังความรู้ 1 รอบ แล้วตอบ"""
    return [
        LLMResult(
            text="",
            usage=usage(100),
            tool_calls=[
                ToolCall(id="c1", name="search_knowledge_base", arguments={"query": "เช็คอิน"})
            ],
        ),
        LLMResult(text=answer, usage=usage(150)),
    ]


def rooms_flow(answer: str) -> list[LLMResult]:
    """โมเดลถามระบบจอง 1 รอบ แล้วตอบ"""
    return [
        LLMResult(
            text="",
            usage=usage(100),
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="search_available_rooms",
                    arguments={"check_in": "2026-09-24", "check_out": "2026-09-25"},
                )
            ],
        ),
        LLMResult(text=answer, usage=usage(150)),
    ]


def test_health():
    client = build_client([LLMResult(text="ok")])
    assert client.get("/api/v1/chatbot/health").json() == {"status": "ok"}


def test_ตอบคำถามนโยบายจากคลังความรู้():
    client = build_client(knowledge_flow("เช็คอินได้ตั้งแต่เวลา 14.00 น. ถึง 22.00 น. ค่ะ"))
    response = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "session-123", "message": "เช็คอินได้กี่โมง", "language": "th"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-123"
    assert "14.00" in data["answer"]
    assert data["answered"] is True
    assert len(data["sources"]) == 1
    assert data["sources"][0]["doc_id"] == "th-booking_policy::0000"
    # token ต้องเป็นผลรวมของทุกรอบในลูป ไม่ใช่แค่รอบสุดท้าย
    assert data["token_usage"]["total_tokens"] == 250


def test_ตอบคำถามห้องว่างด้วยข้อมูลสดจากระบบจอง():
    client = build_client(rooms_flow("พรุ่งนี้มีห้อง P1 ว่างค่ะ ราคา 650 บาทต่อคืน"))
    response = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "พรุ่งนี้มีห้องว่างไหม"},
    )

    data = response.json()
    assert data["answered"] is True
    assert "650" in data["answer"]
    client.mock_booking_client.search_rooms.assert_awaited_once_with(
        check_in="2026-09-24", check_out="2026-09-25"
    )
    assert data["tools_used"] == ["search_available_rooms"]
    # คำตอบที่อิงข้อมูลสดไม่ได้มาจากการค้นคลังความรู้ retrieval_score จึงเป็น 0 ตามจริง
    assert data["retrieval_score"] == 0.0
    # แต่ confidence ต้องไม่ถูกลดตามไปด้วย เพราะข้อมูลจากระบบจองคือข้อมูลจริง
    assert data["confidence"] == 1.0
    assert data["sources"] == []


def test_capacity_คือเพดานที่จองในแอปได้_ไม่มีค่าเตียงเสริมบวกให้():
    """เวอร์ชันก่อนส่ง maxGuestsWithExtraBed = capacity + 1 ไปให้โมเดล

    บอทจึงบอกลูกค้าว่า "จองห้องเดียว 3 คนได้" แต่ POST /api/bookings ตอบ 400
    เมื่อ guests > capacity เพราะระบบจองไม่มีเรื่องเตียงเสริม
    ส่วน amenities ต้องติดไปด้วยเสมอเพราะคลังความรู้ไม่มีรายการของในห้องแล้ว
    """
    from app.services.booking_api import _trim_room

    trimmed = _trim_room(
        {
            "id": "r1",
            "name": "P1",
            "type": "single",
            "pricePerNight": 650,
            "capacity": 2,
            "description": "",
            "amenities": ["Wi-Fi", "Mini fridge"],
        }
    )
    assert trimmed["capacity"] == 2
    assert "maxGuestsWithExtraBed" not in trimmed
    assert trimmed["amenities"] == ["Wi-Fi", "Mini fridge"]


QUOTE_ROOMS = [
    {"id": "1", "name": "P1", "type": "single", "pricePerNight": 650, "capacity": 2,
     "typeLabel": "เตียงเดี่ยว (มีเตียงใหญ่ 1 เตียง)", "description": "", "amenities": []}
]


@pytest.mark.asyncio
async def test_quote_price_คิดเลขให้เสร็จในโค้ด():
    """โมเดลคิดเลขผิดเป็นระยะ เช่น 650 x 1 ห้อง x 2 คืน = 2,600 (ที่ถูกคือ 1,300)

    ผิดแบบนี้ลูกค้าจับไม่ได้เพราะสูตรที่เขียนกำกับถูก ผิดแค่ผลลัพธ์
    """
    from unittest.mock import patch

    from app.services.booking_api import BookingAPIClient

    with patch.object(BookingAPIClient, "search_rooms", return_value=QUOTE_ROOMS):
        quote = await BookingAPIClient().quote_stay("2026-09-24", "2026-09-26")
    assert quote["nights"] == 2          # half-open: เข้า 24 ออก 26 = 2 คืน
    assert quote["total"] == 1300        # ไม่ใช่ 2,600
    # ยอดต้องเป็นค่าห้องอย่างเดียวให้ตรงกับที่แอปเก็บ ไม่มี field เตียงเสริมหลงเหลือ
    assert not any("extraBed" in key for key in quote)
    assert "ไม่รวมเตียงเสริม" in quote["note"]


@pytest.mark.asyncio
async def test_quote_price_ที่โมเดลส่ง_extra_beds_มาต้องถูกเพิกเฉย_ไม่ใช่_error():
    """โมเดลอาจยังจำ schema เก่าแล้วส่ง extra_beds มา ต้องไม่ error และยอดเป็นค่าห้องอย่างเดียว"""
    import json
    from unittest.mock import patch

    from app.core.tools import TOOL_SCHEMAS, ToolExecutor
    from app.services.booking_api import BookingAPIClient

    schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "quote_price")
    assert "extra_beds" not in schema["function"]["parameters"]["properties"]
    assert "extra_bed_price_per_night" not in schema["function"]["parameters"]["properties"]

    executor = ToolExecutor(
        booking_client=BookingAPIClient(),
        embedding_service=MagicMock(),
        vector_store=MagicMock(),
    )
    with patch.object(BookingAPIClient, "search_rooms", return_value=QUOTE_ROOMS):
        raw = await executor.run(
            "quote_price",
            {
                "check_in": "2026-09-24",
                "check_out": "2026-09-26",
                "rooms": 1,
                "extra_beds": 1,
                "extra_bed_price_per_night": 200,
            },
        )
    result = json.loads(raw)
    assert "error" not in result
    assert result["total"] == 1300  # 650 x 2 คืน ไม่บวก 400 ค่าเตียงเสริม
    assert executor.successful_tools == ["quote_price"]


class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def json(self):
        return self._payload


def _capture_httpx(monkeypatch, status_code: int = 200, payload=None) -> list[dict]:
    """แทน httpx.AsyncClient ด้วยตัวปลอมที่บันทึก headers ของทุกคำขอไว้ตรวจ"""
    import app.services.booking_api as booking_api

    captured: list[dict] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, params=None, headers=None):
            captured.append({"url": url, "headers": dict(headers or {})})
            return _FakeResponse(status_code, payload)

    monkeypatch.setattr(booking_api.httpx, "AsyncClient", FakeAsyncClient)
    return captured


@pytest.mark.asyncio
async def test_แนบ_X_Internal_Key_เมื่อตั้งค่าไว้(monkeypatch):
    """NestJS จำกัด 120 ครั้ง/นาที/IP และทุกคำขอของบอทออกจาก IP เดียว จึงต้องแนบ key ทุกครั้ง"""
    from app.config import settings
    from app.services.booking_api import BookingAPIClient

    monkeypatch.setattr(settings, "booking_api_internal_key", "s3cret")
    captured = _capture_httpx(monkeypatch, payload={"holdMinutes": 3})

    await BookingAPIClient().get_payment_info()

    assert captured[0]["headers"]["X-Internal-Key"] == "s3cret"
    # endpoint สาธารณะต้องไม่มี Authorization ติดไป
    assert "Authorization" not in captured[0]["headers"]


@pytest.mark.asyncio
async def test_ไม่แนบ_X_Internal_Key_เมื่อค่าว่าง(monkeypatch):
    from app.config import settings
    from app.services.booking_api import BookingAPIClient

    monkeypatch.setattr(settings, "booking_api_internal_key", "")
    captured = _capture_httpx(monkeypatch, payload=[])

    await BookingAPIClient().list_restaurants()

    assert "X-Internal-Key" not in captured[0]["headers"]


@pytest.mark.asyncio
async def test_get_my_bookings_แนบทั้ง_internal_key_และ_JWT_ของลูกค้า(monkeypatch):
    """internal key แค่ข้าม rate limit ไม่ได้ให้สิทธิ์อ่าน /bookings/me ยังต้องใช้ JWT ของลูกค้า"""
    from app.services.booking_api import BookingAPIClient

    captured = _capture_httpx(monkeypatch, payload=[])

    await BookingAPIClient(internal_key="s3cret").get_my_bookings("customer-jwt")

    headers = captured[0]["headers"]
    assert headers["X-Internal-Key"] == "s3cret"
    assert headers["Authorization"] == "Bearer customer-jwt"


@pytest.mark.asyncio
async def test_429_ถูกแปลงเป็น_BookingAPIError_ที่บอกให้ลองใหม่(monkeypatch):
    from app.services.booking_api import BookingAPIAuthError, BookingAPIClient, BookingAPIError

    _capture_httpx(monkeypatch, status_code=429)

    with pytest.raises(BookingAPIError, match="คำขอมากเกินไป") as exc_info:
        await BookingAPIClient(internal_key="").get_payment_info()
    # ต้องไม่ถูกตีความเป็นเรื่องสิทธิ์ ไม่งั้นบอทจะบอกให้ลูกค้าล็อกอินใหม่
    assert not isinstance(exc_info.value, BookingAPIAuthError)


def test_internal_key_ต้องไม่หลุดไปใน_response(monkeypatch):
    """key อยู่ใน header ขาออกเท่านั้น ห้ามโผล่ใน answer / tools_used / sources"""
    from app.config import settings

    monkeypatch.setattr(settings, "booking_api_internal_key", "do-not-leak-me")
    client = build_client(rooms_flow("มีห้องว่างค่ะ"))
    response = client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": "ว่างไหม"})

    assert "do-not-leak-me" not in response.text
    for call in client.fake_llm.calls:
        assert "do-not-leak-me" not in str(call["messages"])


@pytest.mark.asyncio
async def test_quote_price_ปฏิเสธช่วงวันที่ที่เป็นไปไม่ได้():
    from app.services.booking_api import BookingAPIClient, BookingAPIError

    with pytest.raises(BookingAPIError, match="วันออกต้องอยู่หลังวันเข้าพัก"):
        await BookingAPIClient().quote_stay("2026-09-26", "2026-09-24")


def test_แปลง_tool_call_ที่ปนมาในข้อความให้เป็นการเรียก_tool_จริง():
    """Typhoon บางรอบพิมพ์คำขอเรียก tool ออกมาเป็นข้อความแทนที่จะใช้ field tool_calls

    ถ้าไม่ดัก ลูกค้าจะเห็น JSON ดิบเป็นคำตอบ
    """
    from app.services.llm_client import _extract_inline_tool_calls

    raw = (
        "สักครู่นะคะ\n<tool_call>\n"
        '{"name": "quote_price", "arguments": {"check_in": "2026-09-23", '
        '"check_out": "2026-09-26", "extra_beds": 0}}\n</tool_call>'
    )
    text, calls = _extract_inline_tool_calls(raw)

    assert "<tool_call>" not in text
    assert text == "สักครู่นะคะ"
    assert len(calls) == 1
    assert calls[0].name == "quote_price"
    assert calls[0].arguments["check_out"] == "2026-09-26"


def test_ข้อความปกติต้องไม่ถูกตีความเป็น_tool_call():
    from app.services.llm_client import _extract_inline_tool_calls

    text, calls = _extract_inline_tool_calls("ห้องว่างค่ะ ราคา 650 บาท")
    assert calls == []
    assert text == "ห้องว่างค่ะ ราคา 650 บาท"


def test_ห้ามส่ง_guests_ต่อเข้า_booking_api():
    """การส่ง ?guests=3 จะกรองห้อง capacity 2 ทิ้งหมด ลูกค้าเลยได้ "ไม่มีห้องว่าง"

    ทั้งที่ยังมีทางเลือกคือจองในแอป 2 ท่านแล้วโทรขอเตียงเสริม หรือจอง 2 ห้อง
    ถ้าวันหนึ่งมีคนเติม guests กลับเข้า tool schema test นี้ต้องแดงทันที
    """
    from app.core.tools import TOOL_SCHEMAS

    schema = next(
        tool for tool in TOOL_SCHEMAS if tool["function"]["name"] == "search_available_rooms"
    )
    assert "guests" not in schema["function"]["parameters"]["properties"]

    client = build_client(rooms_flow("มีห้องว่างค่ะ"))
    client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": "มา 3 คน"})
    assert "guests" not in client.mock_booking_client.search_rooms.await_args.kwargs


@pytest.mark.asyncio
async def test_ผลห้องว่างแนบ_note_ให้ไปค้นค่าเตียงเสริม_แต่ไม่มีตัวเลขในโค้ด():
    """ทดสอบจริงโมเดลข้ามการค้นคลังแล้วแต่งค่าเตียงเสริมเอง จึงติด note ไว้กับผล tool

    แต่ตัวเลขค่าเตียงเสริมต้องไม่อยู่ในโค้ด เพราะคลังความรู้เป็นเจ้าของตัวเลขนั้น
    """
    import json
    import re

    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    booking_client.search_rooms.return_value = SAMPLE_ROOMS
    executor = ToolExecutor(booking_client, MagicMock(), MagicMock())

    result = json.loads(
        await executor.run(
            "search_available_rooms", {"check_in": "2026-09-24", "check_out": "2026-09-25"}
        )
    )
    assert result["rooms"] == SAMPLE_ROOMS
    assert "search_knowledge_base" in result["note"]
    assert not re.search(r"\d", result["note"])


def test_กติกาถูกส่งแยกเป็น_system_ไม่ใช่ยัดรวมกับคำถาม():
    """จุดนี้คือสิ่งที่ทำให้โมเดลยอมปฏิเสธแทนที่จะแต่งคำตอบ ถ้าหลุดต้องรู้ทันที"""
    client = build_client(knowledge_flow("เช็คอิน 14.00 น. ค่ะ"))
    client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": "เช็คอินได้กี่โมง"})

    messages = client.fake_llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "ห้ามใช้ความรู้ทั่วไปของคุณเอง" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "เช็คอินได้กี่โมง"
    assert "ห้ามใช้ความรู้ทั่วไปของคุณเอง" not in messages[1]["content"]


def test_system_prompt_ต้องบอกวันที่วันนี้():
    """ถ้าไม่บอก โมเดลจะคิดวันจากช่วงที่มันถูกเทรนมา แล้วยิง tool ด้วยวันที่ในอดีต"""
    import re

    client = build_client(knowledge_flow("ค่ะ"))
    client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": "พรุ่งนี้ว่างไหม"})

    system_prompt = client.fake_llm.calls[0]["messages"][0]["content"]
    assert re.search(r"วันนี้คือวันที่ \d{4}-\d{2}-\d{2}", system_prompt)


def test_ผลของ_tool_ถูกส่งกลับเข้าไปผูกกับ_tool_call_id_เดิม():
    """ถ้า tool_call_id ไม่ตรง โมเดลจะไม่รู้ว่าผลนั้นเป็นคำตอบของคำขอไหน"""
    client = build_client(rooms_flow("มีห้องว่างค่ะ"))
    client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": "พรุ่งนี้ว่างไหม"})

    second_round = client.fake_llm.calls[1]["messages"]
    assert second_round[-2]["role"] == "assistant"
    assert second_round[-2]["tool_calls"][0]["id"] == "c1"
    assert second_round[-1]["role"] == "tool"
    assert second_round[-1]["tool_call_id"] == "c1"
    assert "P1" in second_round[-1]["content"]


def test_ถามการจองของตัวเองโดยไม่ได้ล็อกอินต้องไม่หลุดข้อมูลใคร():
    """ไม่มี token = ไม่เรียก API เลย ไม่ใช่เรียกด้วย token กลางของระบบ"""
    client = build_client(
        [
            LLMResult(
                text="",
                usage=usage(100),
                tool_calls=[ToolCall(id="c1", name="get_my_bookings", arguments={})],
            ),
            LLMResult(text="รบกวนเข้าสู่ระบบก่อนนะคะ", usage=usage(140)),
        ]
    )
    client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": "ฉันจองอะไรไว้บ้าง"})

    client.mock_booking_client.get_my_bookings.assert_not_awaited()
    tool_result = client.fake_llm.calls[1]["messages"][-1]["content"]
    assert "not_authenticated" in tool_result


def test_token_ของลูกค้ามาจาก_header_แล้วถูกส่งต่อให้_booking_api():
    client = build_client(
        [
            LLMResult(
                text="",
                usage=usage(100),
                tool_calls=[ToolCall(id="c1", name="get_my_bookings", arguments={})],
            ),
            LLMResult(text="คุณมีการจอง 1 รายการค่ะ", usage=usage(140)),
        ]
    )
    client.mock_booking_client.get_my_bookings.return_value = []
    client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "ฉันจองอะไรไว้บ้าง"},
        headers={"Authorization": "Bearer customer-token-123"},
    )
    client.mock_booking_client.get_my_bookings.assert_awaited_once_with("customer-token-123")


def _booking(room, check_in, check_out, status, payment="unpaid"):
    return {
        "id": f"{room}-{check_in}",
        "roomName": room,
        "checkIn": check_in,
        "checkOut": check_out,
        "status": status,
        "paymentStatus": payment,
    }


# ข้อมูลชุดเดียวกับที่บอทเคยไล่ตอบมาทั้ง 7 รายการ (วันนี้ = 2026-10-03)
MY_BOOKINGS = [
    _booking("F1", "2026-10-12", "2026-10-18", "approved", "paid"),
    _booking("P1", "2026-09-30", "2026-10-02", "pending"),
    _booking("F1", "2026-09-28", "2026-09-29", "approved"),
    _booking("P2", "2026-09-27", "2026-09-29", "checked_out"),
    _booking("P1", "2026-09-27", "2026-09-28", "checked_out", "paid"),
    _booking("P3", "2026-09-24", "2026-09-26", "approved", "paid"),
    _booking("P4", "2026-10-20", "2026-10-21", "cancelled"),
]


def test_group_bookings_ส่งเฉพาะการจองที่ยังไม่ถึงและนับประวัติแยก():
    from app.services.booking_api import group_bookings

    result = group_bookings(MY_BOOKINGS, today="2026-10-03")

    assert [b["id"] for b in result["currentBookings"]] == ["F1-2026-10-12"]
    assert result["pastBookingsCount"] == 6
    assert "pastBookings" not in result


def test_group_bookings_นับ_checked_in_เป็นปัจจุบัน_และเช็คเอาท์วันนี้เป็นประวัติ():
    from app.services.booking_api import group_bookings

    bookings = [
        _booking("P1", "2026-10-01", "2026-10-03", "checked_in", "paid"),
        _booking("P2", "2026-10-01", "2026-10-03", "approved", "paid"),
        _booking("P3", "2026-10-02", "2026-10-04", "approved", "paid"),
    ]
    result = group_bookings(bookings, today="2026-10-03")

    assert [b["roomName"] for b in result["currentBookings"]] == ["P1", "P3"]
    assert result["pastBookingsCount"] == 1


@pytest.mark.asyncio
async def test_get_my_bookings_include_past_ส่งประวัติมาด้วย():
    import json

    from app.core.tools import TOOL_SCHEMAS, ToolExecutor

    schema = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "get_my_bookings")
    assert "include_past" in schema["function"]["parameters"]["properties"]

    booking_client = MagicMock()
    booking_client.get_my_bookings = AsyncMock(return_value=MY_BOOKINGS)
    executor = ToolExecutor(
        booking_client, MagicMock(), MagicMock(), customer_token="jwt", today="2026-10-03"
    )

    default = json.loads(await executor.run("get_my_bookings", {}))
    with_past = json.loads(await executor.run("get_my_bookings", {"include_past": True}))

    assert len(default["currentBookings"]) == 1 and "pastBookings" not in default
    assert len(with_past["pastBookings"]) == 6
    # ล่าสุดก่อน ให้ "ครั้งที่แล้ว" อยู่บนสุด
    assert with_past["pastBookings"][0]["checkIn"] == "2026-10-20"


def test_ลูปต้องหยุดและบังคับให้สรุปเมื่อโมเดลขอเรียก_tool_ไม่เลิก():
    """โมเดลที่ขอเรียก tool ทุกรอบต้องไม่ทำให้ค่า token บานหรือคืน response เปล่า"""
    endless = [
        LLMResult(
            text="",
            usage=usage(100),
            tool_calls=[ToolCall(id=f"c{i}", name="search_knowledge_base", arguments={"query": "x"})],
        )
        for i in range(10)
    ]
    client = build_client(endless)
    response = client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": "วนไปเรื่อย"})

    data = response.json()
    assert response.status_code == 200
    # รอบสุดท้ายต้องไม่ส่ง tools ไปด้วย เพื่อบังคับให้โมเดลตอบเป็นข้อความ
    assert client.fake_llm.calls[-1]["tools"] is None
    assert data["answer"] == NO_CONTEXT_ANSWER
    assert data["answered"] is False


def test_snippet_ต้องไม่มีบรรทัดบริบทติดไปให้ลูกค้าเห็น():
    client = build_client(knowledge_flow("เช็คอิน 14.00 น. ค่ะ"))
    response = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "เช็คอินได้กี่โมง"}
    )
    snippet = response.json()["sources"][0]["snippet"]
    assert not snippet.startswith("[หมวด:")
    assert "เวลาเช็คอินคือ 14.00 น." in snippet


def test_เมื่อปฏิเสธ_confidence_ต้องเป็นศูนย์และไม่อ้างแหล่งที่มา():
    """confidence 0.84 คู่กับคำตอบ 'ไม่มีข้อมูล' จะทำให้ frontend ตัดสินใจผิด"""
    client = build_client(knowledge_flow(NO_CONTEXT_ANSWER))
    response = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "จากสนามบินหาดใหญ่มากี่กิโล"},
    )

    data = response.json()
    assert data["answered"] is False
    assert data["confidence"] == 0.0
    assert data["sources"] == []
    # แต่คะแนนดิบต้องยังเก็บไว้ให้ debug ได้ว่า retrieval ดึงอะไรมา
    assert data["retrieval_score"] > 0.0
    # token ถูกใช้จริงแม้โมเดลจะปฏิเสธ จึงต้องรายงานด้วย
    assert data["token_usage"]["total_tokens"] > 0


def test_คำตอบที่ตอบได้แล้วแถมประโยคปฏิเสธมาด้วยต้องยังนับว่าตอบได้():
    """โมเดลตอบครบแล้วต่อท้ายด้วยประโยคปฏิเสธ (เจอจริงตอนทดสอบ end-to-end)

    ถ้านับเป็น 'ปฏิเสธ' frontend จะขึ้นปุ่มโทรหารีสอร์ททับคำตอบที่ใช้งานได้จริง
    """
    answer = "ห้องพักมี 2 แบบ คือ single และ twin ค่ะ " + NO_CONTEXT_ANSWER
    client = build_client(knowledge_flow(answer))
    response = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "ห้องพักมีกี่แบบ"},
    )

    data = response.json()
    assert data["answered"] is True
    assert data["confidence"] > 0.0
    assert data["sources"] != []


def test_ปฏิเสธโดยไม่เรียก_tool_เลยต้องถูกกระทุ้งให้ลองใหม่():
    """โมเดลยังไม่ได้ดูข้อมูลอะไรเลย จึงยังไม่มีสิทธิ์สรุปว่าไม่มีข้อมูล"""
    client = build_client(
        [
            LLMResult(text=NO_CONTEXT_ANSWER, usage=usage(100)),
            LLMResult(
                text="",
                usage=usage(120),
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="search_available_rooms",
                        arguments={"check_in": "2026-09-24", "check_out": "2026-09-25"},
                    )
                ],
            ),
            LLMResult(text="พรุ่งนี้มีห้องว่างค่ะ", usage=usage(150)),
        ]
    )
    response = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "เตียงเดี่ยวมีกี่ห้อง"}
    )

    data = response.json()
    assert data["answered"] is True
    assert data["tools_used"] == ["search_available_rooms"]
    # ข้อความกระทุ้งต้องถูกส่งไปจริง ไม่ใช่แค่วนลูปเปล่า
    nudge = client.fake_llm.calls[1]["messages"][-1]
    assert nudge["role"] == "user"
    assert "ยังไม่ได้เรียก tool" in nudge["content"]


def test_กระทุ้งได้ครั้งเดียว_ถ้ายังยืนยันปฏิเสธก็ถือว่าปฏิเสธจริง():
    client = build_client([LLMResult(text=NO_CONTEXT_ANSWER, usage=usage(100))] * 5)
    response = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "ถามอะไรก็ไม่รู้"}
    )
    assert response.json()["answered"] is False
    # 1 รอบแรก + 1 รอบหลังกระทุ้ง = 2 ครั้ง ไม่ใช่วนจนครบ max rounds
    assert len(client.fake_llm.calls) == 2


def test_ปฏิเสธล้วนต้องยังถูกนับว่าปฏิเสธ():
    """เกณฑ์ใหม่ต้องไม่หลวมจนคำตอบที่ปฏิเสธจริงเล็ดลอดไปเป็น answered=true"""
    client = build_client(knowledge_flow(f"  {NO_CONTEXT_ANSWER}  "))
    response = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "ถามอะไรก็ไม่รู้"}
    )
    assert response.json()["answered"] is False


def test_กรอง_chunk_ตามภาษาที่ลูกค้าถาม():
    """ถ้าไม่กรองภาษา ลูกค้าที่ถามไทยอาจได้ chunk อังกฤษกลับไป"""
    client = build_client(knowledge_flow("Check-in is at 14.00."))
    client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "What time is check-in?", "language": "en"},
    )
    assert client.mock_vector_store.query.call_args.kwargs["language"] == "en"


def test_ภาษาที่ไม่รองรับถูกแทนด้วยค่าเริ่มต้น():
    client = build_client(knowledge_flow("ค่ะ"))
    client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": "test", "language": "ja"})
    assert client.mock_vector_store.query.call_args.kwargs["language"] == "th"


def test_ส่ง_resort_id_มาก็ยังรับได้_แต่ไม่พัง():
    """client เดิมที่ยังส่ง resort_id มาต้องไม่ error"""
    client = build_client(knowledge_flow("ค่ะ"))
    response = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "resort_id": "sunrise_villa", "message": "เช็คอินได้กี่โมง"},
    )
    assert response.status_code == 200


def test_พูดถึงเตียงเสริมโดยไม่ได้ค้นคลังต้องถูกกระทุ้งให้ไปค้นก่อน():
    """ทดสอบจริงโมเดลตอบ "เตียงเสริมคืนละ 300 บาท" โดยไม่เรียก search_knowledge_base เลย"""
    guessed = "พักห้องเดียวได้ค่ะ จองในแอป 2 ท่าน แล้วโทรขอเตียงเสริม คืนละ 300 บาท"
    corrected = "พักห้องเดียวได้ค่ะ จองในแอป 2 ท่าน แล้วโทรขอเตียงเสริม คืนละ 200 บาท"
    client = build_client(
        rooms_flow(guessed)
        + [
            LLMResult(
                text="",
                usage=usage(100),
                tool_calls=[
                    ToolCall(id="c2", name="search_knowledge_base", arguments={"query": "เตียงเสริม"})
                ],
            ),
            LLMResult(text=corrected, usage=usage(150)),
        ]
    )
    data = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "มา 3 คน พักห้องเดียวได้ไหม"}
    ).json()

    assert data["answer"] == corrected
    assert data["tools_used"] == ["search_available_rooms", "search_knowledge_base"]


def test_พูดถึงเตียงเสริมหลังค้นคลังแล้วต้องไม่ถูกกระทุ้งซ้ำ():
    answer = "เตียงเสริมคืนละ 200 บาทต่อเตียง ชำระที่รีสอร์ทวันเช็คอินค่ะ"
    client = build_client(knowledge_flow(answer))
    data = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "เตียงเสริมกี่บาท"}
    ).json()

    assert data["answer"] == answer
    assert len(client.fake_llm.calls) == 2  # ค้น 1 รอบ + ตอบ 1 รอบ ไม่มีรอบกระทุ้ง
