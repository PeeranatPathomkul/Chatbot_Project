"""
Test สำหรับ endpoint /api/v1/chatbot/query และ /api/v1/chatbot/health

หมายเหตุ: mock LLMClient, BookingAPIClient, VectorStore และ EmbeddingService ไว้ทั้งหมด
เพื่อไม่ให้ยิง API จริงหรือโหลดโมเดล embedding จริงตอนรัน test (ประหยัดเวลา/เงิน)

**FakeLLMClient คืนผลลัพธ์เป็นคิว** เพราะการตอบหนึ่งคำถามไม่ใช่การเรียก LLM ครั้งเดียว
อีกต่อไป แต่เป็นลูป: ขอเรียก tool -> รับผล -> ตอบ การ mock ด้วยค่าคืนค่าเดียว
จะทำให้ลูปวนไม่รู้จบเพราะโมเดลจำลองขอเรียก tool ซ้ำทุกครั้ง
"""

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.prompts import NO_CONTEXT_ANSWER
from app.main import app
from app.services.chat_history import ChatHistoryStore
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


def temp_history_store(**kwargs) -> ChatHistoryStore:
    """ที่เก็บประวัติแยกไฟล์ต่อ test กันไม่ให้ test เขียนลง data/chat_history.db ตัวจริง
    และกันประวัติของ test หนึ่งรั่วไปโผล่ใน prompt ของอีก test"""
    return ChatHistoryStore(path=Path(tempfile.mkdtemp()) / "chat_history.db", **kwargs)


def build_pipeline(
    llm_results: list[LLMResult],
    hits: list[dict] | None = None,
    history_store: ChatHistoryStore | None = None,
):
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
        history_store=history_store or temp_history_store(),
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
    """โมเดลถามระบบจอง 1 รอบ แล้วตอบ

    วันที่นับจากวันนี้ เพราะ pipeline ใช้วันนี้จริงและปัดวันในอดีตทิ้งก่อนถึงระบบจอง
    """
    check_in = date.today() + timedelta(days=7)
    return [
        LLMResult(
            text="",
            usage=usage(100),
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="search_available_rooms",
                    arguments={
                        "check_in": check_in.isoformat(),
                        "check_out": (check_in + timedelta(days=1)).isoformat(),
                    },
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
    args = rooms_flow("")[0].tool_calls[0].arguments
    client.mock_booking_client.search_rooms.assert_awaited_once_with(
        check_in=args["check_in"], check_out=args["check_out"]
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
        today="2026-09-20",
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
    booking_client.list_rooms.return_value = SAMPLE_ROOMS + [{**SAMPLE_ROOMS[0], "name": "P2"}]
    executor = ToolExecutor(booking_client, MagicMock(), MagicMock(), today="2026-09-20")

    result = json.loads(
        await executor.run(
            "search_available_rooms", {"check_in": "2026-09-24", "check_out": "2026-09-25"}
        )
    )
    assert [room["name"] for room in result["rooms"]] == ["P1"]
    # amenities แปลเป็นคำที่ลูกค้าพิมพ์จริง (ถาม "ตู้เย็น" ต้องจับคู่ได้ ไม่ใช่ "Mini fridge")
    assert result["rooms"][0]["amenities"] == ["Wi-Fi", "เครื่องปรับอากาศ"]
    # ห้องที่มีอยู่แต่ไม่ว่าง ต้องบอกให้ชัด ไม่งั้นบอทตอบว่า "ไม่มีห้องนี้ในระบบ"
    assert result["unavailableRoomNames"] == ["P2"]
    assert result["dateRangeLabel"] == "เข้าพัก 24 ก.ย. 2569 ออก 25 ก.ย. 2569"
    # ให้ไปคิดจำนวนห้อง/เตียงเสริมที่ quote_price ซึ่งอ่านค่าเตียงเสริมจากคลังความรู้เอง
    assert "quote_price" in result["note"]
    assert not re.search(r"\d", result["note"])


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["search_available_rooms", "quote_price"])
@pytest.mark.parametrize(
    ("check_in", "check_out", "error"),
    [
        # เจอจริง: ลูกค้าถาม "29 ตุลา 2004" บอทเปลี่ยนเป็น 2569 เองแล้วตอบว่าห้องว่าง
        ("2004-10-29", "2004-10-30", "date_in_past"),
        ("2026-10-06", "2026-10-08", "date_in_past"),
        # ปี พ.ศ. ที่หลุดมาเป็น ค.ศ.
        ("2569-10-29", "2569-10-30", "date_looks_buddhist"),
        ("2026-02-31", "2026-03-01", "bad_arguments"),
        ("2026-10-09", "2026-10-08", "bad_arguments"),
        ("29/10/2026", "30/10/2026", "bad_arguments"),
    ],
)
async def test_วันที่ใช้ไม่ได้ต้องไม่ถึง_booking_api(tool, check_in, check_out, error):
    """Booking API ตอบห้องว่างให้วันในอดีตตามปกติ จึงต้องกันไว้ที่ฝั่งบอท"""
    import json

    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    executor = ToolExecutor(booking_client, MagicMock(), MagicMock(), today="2026-10-07")

    result = json.loads(
        await executor.run(tool, {"check_in": check_in, "check_out": check_out})
    )
    assert result["error"] == error
    booking_client.search_rooms.assert_not_awaited()
    booking_client.quote_stay.assert_not_awaited()
    assert executor.successful_tools == []


@pytest.mark.asyncio
async def test_เข้าพักวันนี้ยังค้นได้():
    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    booking_client.search_rooms.return_value = SAMPLE_ROOMS
    executor = ToolExecutor(booking_client, MagicMock(), MagicMock(), today="2026-10-07")

    await executor.run(
        "search_available_rooms", {"check_in": "2026-10-07", "check_out": "2026-10-08"}
    )
    assert executor.successful_tools == ["search_available_rooms"]


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


def knowledge_refusal_flow() -> list[LLMResult]:
    """ค้นคลังความรู้แล้วปฏิเสธ ถูกกระทุ้งให้ตรวจซ้ำ แล้วยืนยันปฏิเสธ"""
    return knowledge_flow(NO_CONTEXT_ANSWER) + [LLMResult(text=NO_CONTEXT_ANSWER, usage=usage(100))]


def test_เมื่อปฏิเสธ_confidence_ต้องเป็นศูนย์และไม่อ้างแหล่งที่มา():
    """confidence 0.84 คู่กับคำตอบ 'ไม่มีข้อมูล' จะทำให้ frontend ตัดสินใจผิด"""
    client = build_client(knowledge_refusal_flow())
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


def test_กระทุ้งแต่ละแบบได้ครั้งเดียว_ถ้ายังยืนยันปฏิเสธก็ถือว่าปฏิเสธจริง():
    client = build_client([LLMResult(text=NO_CONTEXT_ANSWER, usage=usage(100))] * 8)
    response = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "ถามอะไรก็ไม่รู้"}
    )
    assert response.json()["answered"] is False
    # รอบแรก -> กระทุ้งให้เรียก tool -> ค้นคลังความรู้ให้เอง -> กระทุ้งให้ตรวจผลซ้ำ แล้วหยุด
    # ไม่ใช่วนจนครบจำนวนรอบสูงสุด
    assert len(client.fake_llm.calls) == 4


def test_ปฏิเสธล้วนต้องยังถูกนับว่าปฏิเสธ():
    """เกณฑ์ใหม่ต้องไม่หลวมจนคำตอบที่ปฏิเสธจริงเล็ดลอดไปเป็น answered=true"""
    client = build_client(
        knowledge_flow(f"  {NO_CONTEXT_ANSWER}  ")
        + [LLMResult(text=f"  {NO_CONTEXT_ANSWER}  ", usage=usage(100))]
    )
    response = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "ถามอะไรก็ไม่รู้"}
    )
    assert response.json()["answered"] is False


POOL_HITS = [
    {
        "doc_id": "th-faq::0010",
        "text": "ถาม: มีสระว่ายน้ำไหม\nตอบ: ไม่มีสระว่ายน้ำ รีสอร์ทเป็นที่พักขนาดเล็ก\n"
                "ซักรีด: <<รอเติม: มีบริการซักรีดหรือไม่>>\n"
                "เงินประกันความเสียหาย: ยังไม่ได้รับข้อมูลจากรีสอร์ท ไม่มีข้อมูลเรื่องนี้ในระบบ\n"
                "รับฝากกระเป๋า: รีสอร์ทไม่มีบริการรับฝากกระเป๋า",
        "metadata": {"source": "th/faq.md", "language": "th"},
        "distance": 0.15,
    }
]


def test_ปฏิเสธหลังค้นแล้วต้องถูกกระทุ้งให้ตรวจผล_tool_ซ้ำ():
    """ทดสอบจริง 'มีสระว่ายน้ำไหม' ค้นเจอ 'ไม่มีสระว่ายน้ำ' แล้วยังตอบว่าไม่มีข้อมูล"""
    client = build_client(
        knowledge_flow(NO_CONTEXT_ANSWER)
        + [LLMResult(text="ไม่มีสระว่ายน้ำค่ะ\nอ้างอิง: ไม่มีสระว่ายน้ำ", usage=usage(100))],
        hits=POOL_HITS,
    )
    response = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "มีสระว่ายน้ำไหม"}
    )
    data = response.json()
    # บรรทัดอ้างอิงมีไว้ให้โค้ดตรวจ ไม่ต้องให้ลูกค้าเห็น
    assert data["answer"] == "ไม่มีสระว่ายน้ำค่ะ"
    assert data["answered"] is True
    assert "'ไม่มี'" in client.fake_llm.calls[2]["messages"][-1]["content"]


@pytest.mark.parametrize(
    ("question", "retry"),
    [
        # ไม่ยกประโยคมาอ้าง
        ("มีสระว่ายน้ำไหม", "ไม่มีสระว่ายน้ำค่ะ"),
        # ยก <<รอเติม>> มาอ้าง (เคยตอบว่า "ไม่มีบริการซักผ้า" ทั้งที่ยังไม่มีข้อมูล)
        ("มีบริการซักผ้าไหม", "ไม่มีบริการซักผ้าค่ะ\nอ้างอิง: ซักรีด: <<รอเติม: มีบริการซักรีดหรือไม่>>"),
        # ยกประโยคปฏิเสธเรื่องอื่นมาอ้าง
        ("มีบริการซักผ้าไหม", "ไม่มีบริการซักผ้าค่ะ\nอ้างอิง: รีสอร์ทไม่มีบริการรับฝากกระเป๋า"),
        # แต่งประโยคอ้างอิงที่ไม่มีอยู่ในผล tool
        ("มีเครื่องทำน้ำอุ่นไหม", "ไม่มีค่ะ\nอ้างอิง: ไม่มีเครื่องทำน้ำอุ่น"),
        # รูปที่ indexing-pipeline ใช้แทน <<รอเติม>> จริง ๆ ในคลังความรู้
        ("มีเงินประกันความเสียหายไหม",
         "ไม่มีค่ะ\nอ้างอิง: เงินประกันความเสียหาย: ยังไม่ได้รับข้อมูลจากรีสอร์ท"),
    ],
)
def test_ตรวจซ้ำแล้วอ้างอิงไม่ผ่าน_ต้องกลับไปปฏิเสธ(question, retry):
    """เวอร์ชันแรกของการกระทุ้งทำให้บอทตอบ 'ไม่มี' กับเรื่องที่ยังเป็น <<รอเติม>>"""
    client = build_client(
        knowledge_flow(NO_CONTEXT_ANSWER) + [LLMResult(text=retry, usage=usage(100))],
        hits=POOL_HITS,
    )
    data = client.post("/api/v1/chatbot/query", json={"session_id": "s1", "message": question}).json()
    assert data["answer"] == NO_CONTEXT_ANSWER
    assert data["answered"] is False


def test_ปฏิเสธโดยไม่เรียก_tool_สองรอบ_โค้ดค้นคลังความรู้ให้เอง():
    """ทดสอบจริง 'ขอเบอร์โทร' 'มี LINE ไหม' กระทุ้งแล้วก็ยังปฏิเสธโดยไม่เรียก tool"""
    client = build_client(
        [
            LLMResult(text=NO_CONTEXT_ANSWER, usage=usage(100)),
            LLMResult(text=NO_CONTEXT_ANSWER, usage=usage(100)),
            LLMResult(text="เช็คอินได้ถึง 22.00 น. ค่ะ", usage=usage(100)),
        ]
    )
    data = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "ถึงตีหนึ่งได้ไหม"}
    ).json()
    assert data["answer"] == "เช็คอินได้ถึง 22.00 น. ค่ะ"
    assert data["tools_used"] == ["search_knowledge_base"]
    tool_message = client.fake_llm.calls[2]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "auto_search"


def test_ชื่อห้องที่แต่งขึ้นเองต้องถูกกระทุ้ง():
    """เคยตอบว่า 'ห้องที่ไม่ว่างคือ P5 และ P6' ทั้งที่ไม่มีห้องชื่อนี้"""
    client = build_client(
        rooms_flow("ว่างทุกห้องค่ะ ห้องที่ไม่ว่างคือ P5 และ P6")
        + [LLMResult(text="ห้อง P1 ว่างค่ะ", usage=usage(100))]
    )
    client.mock_booking_client.list_rooms.return_value = SAMPLE_ROOMS
    data = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "สุดสัปดาห์ถัดไปว่างไหม"}
    ).json()
    assert data["answer"] == "ห้อง P1 ว่างค่ะ"
    assert "P5, P6" in client.fake_llm.calls[2]["messages"][-1]["content"]


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
    client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "เช็คอินกี่โมง", "language": "ja"},
    )
    assert client.mock_vector_store.query.call_args.kwargs["language"] == "th"


@pytest.mark.parametrize(
    ("message", "requested", "expected"),
    [
        # แอปตั้งไทย ลูกค้าขอให้ตอบอังกฤษ (เคยตอบว่า "I can only communicate in Thai")
        ("Sorry, can you answer in English?", "th", "en"),
        # แอปตั้งอังกฤษ ลูกค้าพิมพ์ไทย (เคยตอบอังกฤษและค้นคลังฝั่งอังกฤษจนหาไม่เจอ)
        ("มีที่จอดรถไหม", "en", "th"),
        ("Ada bilik kosong esok?", "th", "en"),
        # ไม่มีทั้งไทยและละติน ใช้ค่าที่แอปส่งมา
        ("明天有空房吗", "th", "th"),
        ("👍", "en", "en"),
    ],
)
def test_ตอบตามภาษาที่ลูกค้าพิมพ์จริง(message, requested, expected):
    from app.services.rag_pipeline import _detect_language

    assert _detect_language(message, requested) == expected


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


# ---------------------------------------------------------------------------
# ประวัติบทสนทนา
# ---------------------------------------------------------------------------


class FakeClock:
    """นาฬิกาที่เดินเองได้ ใช้จำลองเวลาผ่านไปหลายวันโดยไม่ต้องรอจริง"""

    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_คำถามถัดไปเห็นคู่ถาม_ตอบก่อนหน้าใน_session_เดียวกัน():
    store = temp_history_store()
    first, _, _, _ = build_pipeline(knowledge_flow("เช็คอิน 14.00 น. ค่ะ"), history_store=store)
    app.dependency_overrides[get_rag_pipeline] = lambda: first
    TestClient(app).post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "เช็คอินได้กี่โมง"}
    )

    second, fake_llm, _, _ = build_pipeline(
        knowledge_flow("เช็คเอาท์ 12.00 น. ค่ะ"), history_store=store
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: second
    TestClient(app).post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "แล้วเช็คเอาท์ล่ะ"}
    )

    messages = fake_llm.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "เช็คอินได้กี่โมง"
    assert messages[2]["content"] == "เช็คอิน 14.00 น. ค่ะ"
    assert messages[3]["content"] == "แล้วเช็คเอาท์ล่ะ"


def test_ประวัติไม่เก็บผลของ_tool_ของรอบก่อน():
    """ผล tool กิน token มากที่สุด และห้องว่างของเมื่อวานไม่ใช่ห้องว่างของวันนี้"""
    store = temp_history_store()
    pipeline, _, _, _ = build_pipeline(rooms_flow("มีห้อง P1 ว่างค่ะ"), history_store=store)
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    TestClient(app).post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "พรุ่งนี้ว่างไหม"}
    )

    history = store.load("s1")
    assert history == [
        {"role": "user", "content": "พรุ่งนี้ว่างไหม"},
        {"role": "assistant", "content": "มีห้อง P1 ว่างค่ะ"},
    ]


def test_ประวัติไม่เก็บข้อความกระทุ้งและคำตอบที่ถูกกระทุ้งให้ตอบใหม่():
    store = temp_history_store()
    pipeline, _, _, _ = build_pipeline(
        [
            LLMResult(text=NO_CONTEXT_ANSWER, usage=usage(100)),
            *rooms_flow("พรุ่งนี้มีห้องว่างค่ะ"),
        ],
        history_store=store,
    )
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    TestClient(app).post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "เตียงเดี่ยวมีกี่ห้อง"}
    )

    assert store.load("s1") == [
        {"role": "user", "content": "เตียงเดี่ยวมีกี่ห้อง"},
        {"role": "assistant", "content": "พรุ่งนี้มีห้องว่างค่ะ"},
    ]


def test_ส่งประวัติให้_LLM_แค่_max_messages_ข้อความล่าสุด():
    store = temp_history_store(max_messages=10)
    for i in range(6):
        store.append("s1", f"คำถาม {i}", f"คำตอบ {i}")

    history = store.load("s1")
    assert len(history) == 10
    # คู่แรกสุด (คำถาม 0) ถูกตัดทิ้ง เหลือ 5 คู่ล่าสุด
    assert history[0] == {"role": "user", "content": "คำถาม 1"}
    assert history[-1] == {"role": "assistant", "content": "คำตอบ 5"}


def test_ประวัติต้องขึ้นต้นด้วยฝั่งลูกค้าเสมอแม้ตั้ง_max_messages_เป็นเลขคี่():
    store = temp_history_store(max_messages=3)
    store.append("s1", "คำถาม 0", "คำตอบ 0")
    store.append("s1", "คำถาม 1", "คำตอบ 1")

    history = store.load("s1")
    assert history[0]["role"] == "user"
    assert [m["content"] for m in history] == ["คำถาม 1", "คำตอบ 1"]


def test_ประวัติไม่ปนกันข้าม_session():
    store = temp_history_store()
    store.append("session-a", "การจองของฉันมีอะไรบ้าง", "คุณมีการจองห้อง P1 ค่ะ")

    assert store.load("session-b") == []


def test_ประวัติที่เก่ากว่า_7_วันไม่ถูกส่งให้_LLM_และถูกลบทิ้ง():
    clock = FakeClock()
    store = temp_history_store(ttl_days=7, clock=clock)
    store.append("old-session", "คำถามเก่า", "คำตอบเก่า")

    clock.now += 6 * 24 * 60 * 60
    assert len(store.load("old-session")) == 2  # ยังไม่ครบ 7 วัน ต้องยังอยู่

    clock.now += 2 * 24 * 60 * 60
    # ยังไม่มีใคร append ให้เกิดการลบ แต่ load ต้องกรองของหมดอายุออกเอง
    assert store.load("old-session") == []

    # append ของ session อื่นต้องกวาดของหมดอายุทิ้งจากไฟล์จริง ไม่ใช่แค่ซ่อนไว้
    store.append("new-session", "คำถามใหม่", "คำตอบใหม่")
    with store._connect() as conn:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id = 'old-session'"
        ).fetchone()
    assert count == 0


# ---------------------------------------------------------------------------
# รอบทดสอบ 134 เคส (ต.ค. 2569): แก้ในโค้ดสิ่งที่โมเดลทำพลาดซ้ำ ๆ
# ---------------------------------------------------------------------------

FIVE_ROOMS = [
    {"id": str(i), "name": name, "type": "single", "pricePerNight": 650, "capacity": 2,
     "typeLabel": "เตียงเดี่ยว", "description": "", "amenities": []}
    for i, name in enumerate(["F1", "P1", "P2", "P3", "P4"])
]


@pytest.mark.parametrize(
    ("guests", "one_room", "one_room_bed", "options"),
    [
        # (จำนวนห้อง, เตียงเสริม, จ่ายในแอป, จ่ายที่รีสอร์ท, รวม) 1 คืน ค่าเตียงเสริม 200
        (3, False, True, [(2, 0, 1300, 0, 1300), (1, 1, 650, 200, 850)]),
        # เคยตอบ "ห้องเดียวขอเตียงเสริมได้ค่ะ" ทั้งที่ห้องเดียวรับได้แค่ 3 คน
        (4, False, False, [(2, 0, 1300, 0, 1300)]),
        # เคยแนะนำแค่ 3 ห้อง ทั้งที่ 2 ห้อง + เตียงเสริม 2 เตียงถูกกว่า
        (6, False, False, [(3, 0, 1950, 0, 1950), (2, 2, 1300, 400, 1700)]),
        # เคยตอบ "ไม่มีข้อมูล" ทั้งที่ห้องว่าง 5 ห้องพอดี
        (10, False, False, [(5, 0, 3250, 0, 3250), (4, 2, 2600, 400, 3000)]),
    ],
)
def test_คิดจำนวนห้องกับเตียงเสริมในโค้ด(guests, one_room, one_room_bed, options):
    from app.services.booking_api import plan_rooms_for_guests

    plan = plan_rooms_for_guests(FIVE_ROOMS, guests=guests, nights=1, extra_bed_rate=200)
    assert plan["fitsInOneRoom"] is one_room
    assert plan["fitsInOneRoomWithExtraBed"] is one_room_bed
    got = [
        (o["rooms"], o["extraBeds"], o["payInApp"], o["payAtResortForExtraBeds"], o["grandTotal"])
        for o in plan["options"]
    ]
    assert got == options


def test_ห้องไม่พอแม้ขอเตียงเสริม_ต้องบอกว่าห้องไม่พอ():
    from app.services.booking_api import plan_rooms_for_guests

    plan = plan_rooms_for_guests(FIVE_ROOMS[:2], guests=7, nights=1, extra_bed_rate=200)
    assert plan["enoughRoomsAvailable"] is False
    assert plan["options"] == []


def test_ไม่รู้ค่าเตียงเสริม_ห้ามเดายอดรวม():
    from app.services.booking_api import plan_rooms_for_guests

    plan = plan_rooms_for_guests(FIVE_ROOMS, guests=3, nights=2, extra_bed_rate=None)
    bed_option = plan["options"][1]
    assert bed_option["payAtResortForExtraBeds"] is None
    assert bed_option["grandTotal"] is None


@pytest.mark.asyncio
async def test_quote_price_ที่ใส่_guests_อ่านค่าเตียงเสริมจากคลังความรู้():
    import json

    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    booking_client.search_rooms.return_value = FIVE_ROOMS
    vector_store = MagicMock()
    vector_store.query.return_value = [
        {"doc_id": "th-booking_policy::0002", "distance": 0.1,
         "text": "แล้วโทรแจ้งเจ้าของเพื่อขอเตียงเสริม (ห้องละไม่เกิน 1 เตียง คืนละ 200 บาทต่อเตียง)"}
    ]
    executor = ToolExecutor(
        booking_client, MagicMock(), vector_store, today="2026-10-08", guests_hint=3
    )

    result = json.loads(await executor.run(
        "quote_price", {"check_in": "2026-10-25", "check_out": "2026-10-27", "guests": 3}
    ))
    assert result["extraBedRatePerNight"] == 200
    assert result["options"][1]["grandTotal"] == 650 * 2 + 200 * 2
    assert executor.planned_guests == [3]
    # แหล่งที่มาของค่าเตียงเสริมต้องโผล่ใน sources ด้วย
    assert executor.knowledge_hits[0]["doc_id"] == "th-booking_policy::0002"


def test_นับครั้งที่เคยพักแยกจากรายการที่ยกเลิก():
    from app.services.booking_api import group_bookings

    bookings = [{**b, "customerName": "Jane Customer"} for b in MY_BOOKINGS]
    result = group_bookings(bookings, today="2026-10-03")
    assert result["customerName"] == "Jane Customer"
    assert result["stayedCount"] == 2
    assert result["cancelledCount"] == 1
    assert result["pastBookingsCount"] == 6
    assert len(result["recentPastBookings"]) == 5
    assert all("customerName" not in b for b in result["currentBookings"])


@pytest.mark.parametrize(
    ("message", "check_in", "error"),
    [
        # เคยเปลี่ยนปีเองเงียบ ๆ ก่อนเรียก tool
        ("29 ตุลา 2004 ว่างไหม", "2026-10-29", "year_mismatch"),
        ("29 ต.ค. 2569 ว่างไหม", "2026-10-29", None),
        ("สงกรานต์ 13-15 เมษา 2027", "2027-04-13", None),
        # เลข 4 หลักใน booking id ต้องไม่ถูกนับเป็นปี
        ("การจอง 2e76d255-0292-40cd-89eb-b42290652e22 ว่างไหม", "2026-10-29", None),
        # "ตุลาคมปีหน้า" เคยได้ปีนี้ 1 ใน 2 รอบ
        ("ตุลาคมปีหน้า วันที่ 20 ว่างไหม", "2026-10-20", "year_mismatch"),
        ("ตุลาคมปีหน้า วันที่ 20 ว่างไหม", "2027-10-20", None),
        # "10/11" แบบไทยคือ 10 พ.ย. เคยถูกอ่านเป็น 10-11 ต.ค.
        ("10/11 ว่างไหม", "2026-10-10", "date_format_mismatch"),
        ("10/11 ว่างไหม", "2026-11-10", None),
        # "เมื่อวาน" เคยถูกค้นเป็นวันนี้แล้วตอบว่ามีห้องว่าง
        ("เมื่อวานมีห้องว่างไหม", "2026-10-08", "date_in_past"),
    ],
)
@pytest.mark.asyncio
async def test_ปีที่ส่งเข้า_tool_ต้องตรงกับปีที่ลูกค้าพิมพ์(message, check_in, error):
    import json

    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    booking_client.search_rooms.return_value = SAMPLE_ROOMS
    executor = ToolExecutor(
        booking_client, MagicMock(), MagicMock(), today="2026-10-08", user_message=message
    )
    check_out = (date.fromisoformat(check_in) + timedelta(days=1)).isoformat()
    result = json.loads(await executor.run(
        "search_available_rooms", {"check_in": check_in, "check_out": check_out}
    ))
    assert result.get("error") == error


@pytest.mark.asyncio
async def test_เจอวันที่ใช้ไม่ได้แล้วห้ามหนีไปค้นวันนี้แทน():
    import json

    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    booking_client.search_rooms.return_value = SAMPLE_ROOMS
    executor = ToolExecutor(
        booking_client, MagicMock(), MagicMock(), today="2026-10-08", user_message="5 ตุลา ว่างไหม"
    )
    first = json.loads(await executor.run(
        "search_available_rooms", {"check_in": "2026-10-05", "check_out": "2026-10-06"}
    ))
    second = json.loads(await executor.run(
        "search_available_rooms", {"check_in": "2026-10-08", "check_out": "2026-10-09"}
    ))
    assert first["error"] == "date_in_past"
    assert second["error"] == "must_ask_customer"
    booking_client.search_rooms.assert_not_awaited()


def _payment_executor(user_message=""):
    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    booking_client.get_my_bookings.return_value = [
        {**_booking("F1", "2026-10-12", "2026-10-18", "approved", "paid"),
         "id": "57e4ad8b-7f4d-4757-b94f-50eb9a154fc0", "totalPrice": 3900},
        {**_booking("P3", "2026-09-24", "2026-09-26", "approved", "paid"),
         "id": "2e76d255-0292-40cd-89eb-b42290652e22", "totalPrice": 1300},
    ]
    booking_client.get_booking_payment.side_effect = lambda booking_id, token: {
        "bookingId": booking_id, "status": "succeeded"
    }
    executor = ToolExecutor(
        booking_client, MagicMock(), MagicMock(),
        customer_token="jwt", today="2026-10-08", user_message=user_message,
    )
    return executor, booking_client


@pytest.mark.asyncio
async def test_ถามจ่ายแล้วยังโดยไม่ระบุเลข_ดูการจองที่ยังไม่ถึงให้():
    """'การจองของฉัน' -> 'อันนั้นจ่ายแล้วยัง' เคยได้ 'ไม่มีข้อมูล' เพราะโมเดลแต่ง id"""
    import json

    executor, _ = _payment_executor("อันนั้นจ่ายแล้วยัง")
    result = json.loads(await executor.run("get_booking_payment", {}))
    assert [b["booking"]["roomName"] for b in result["bookings"]] == ["F1"]
    assert result["bookings"][0]["payment"]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_ใช้เลขการจองที่ลูกค้าพิมพ์_ไม่ใช่เลขที่โมเดลส่งมา():
    """เคยถามสถานะของ 2e76d255… แล้วบอทตอบของ 57e4ad8b…"""
    import json

    executor, _ = _payment_executor(
        "การจองเลขที่ 2e76d255-0292-40cd-89eb-b42290652e22 สถานะการชำระเงินเป็นยังไง"
    )
    result = json.loads(await executor.run(
        "get_booking_payment", {"booking_id": "57e4ad8b-7f4d-4757-b94f-50eb9a154fc0"}
    ))
    assert result["bookings"][0]["booking"]["id"] == "2e76d255-0292-40cd-89eb-b42290652e22"


@pytest.mark.asyncio
async def test_เลขการจองของคนอื่น_ตอบว่าไม่พบ_และไม่ยิงไปถาม_backend():
    import json

    other = "66c78155-9baf-4475-bf4b-fbbab5b8cd67"
    executor, booking_client = _payment_executor(f"การจองเลขที่ {other} จ่ายแล้วหรือยัง")
    result = json.loads(await executor.run("get_booking_payment", {"booking_id": other}))
    assert result["error"] == "booking_not_found"
    booking_client.get_booking_payment.assert_not_called()


@pytest.mark.asyncio
async def test_การจองของฉันมีรายละเอียดห้องที่จองไว้():
    """'ห้องที่จองไว้มีตู้เย็นไหม' เคยถามวันที่กลับ และ 'เตียงอะไร' เคยตอบชื่อห้องผิด"""
    import json

    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    booking_client.get_my_bookings.return_value = [
        {**_booking("F1", "2026-10-12", "2026-10-18", "approved", "paid"), "roomId": "room-f1"}
    ]
    booking_client.get_room.return_value = {
        **SAMPLE_ROOMS[0], "name": "F1", "typeLabel": "เตียงเดี่ยว (มีเตียงใหญ่ 1 เตียง)",
        "amenities": ["Wi-Fi", "TV", "Coffee"],
    }
    executor = ToolExecutor(
        booking_client, MagicMock(), MagicMock(), customer_token="jwt", today="2026-10-08"
    )
    result = json.loads(await executor.run("get_my_bookings", {}))
    booking = result["currentBookings"][0]
    assert booking["roomTypeLabel"] == "เตียงเดี่ยว (มีเตียงใหญ่ 1 เตียง)"
    assert booking["roomAmenities"] == ["Wi-Fi", "ทีวี", "ชุดชงกาแฟ"]
    assert booking["dateRangeLabel"] == "เข้าพัก 12 ต.ค. 2569 ออก 18 ต.ค. 2569"
    assert "roomId" not in booking


def test_ปฏิทินบอกเสาร์อาทิตย์นี้กับหน้าให้ชัด():
    """'เสาร์อาทิตย์หน้า' เคยได้ช่วงของสุดสัปดาห์นี้"""
    from app.services.rag_pipeline import _calendar

    calendar = _calendar("2026-10-08")  # วันพฤหัสบดี
    assert "- 2026-10-08 วันพฤหัสบดี (วันนี้)" in calendar
    assert "- 2026-10-09 วันศุกร์ (พรุ่งนี้)" in calendar
    assert "เสาร์อาทิตย์นี้ = เข้า 2026-10-10 ออก 2026-10-12" in calendar
    assert "เสาร์อาทิตย์หน้า = เข้า 2026-10-17 ออก 2026-10-19" in calendar


def test_คำตอบภาษาจีนต้องถูกกระทุ้งให้ตอบใหม่():
    client = build_client(
        [
            LLMResult(text="请告知入住日期", usage=usage(100)),
            LLMResult(text="", usage=usage(100), tool_calls=[
                ToolCall(id="c1", name="search_knowledge_base", arguments={"query": "เด็ก"})
            ]),
            LLMResult(text="เด็กพักได้ค่ะ", usage=usage(100)),
        ]
    )
    data = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "ผู้ใหญ่ 2 เด็ก 2 ห้องเดียวพอไหม"},
    ).json()
    assert "请" not in data["answer"]
    assert "ภาษาไทย" in client.fake_llm.calls[1]["messages"][-1]["content"]


def test_ลิงก์ที่แต่งเองต้องถูกตัดทิ้ง():
    """เคยแต่งลิงก์ Google Maps ขึ้นมาเองทั้งที่คลังความรู้ยังไม่มีลิงก์"""
    fake = "รีสอร์ทอยู่สะเดาค่ะ ดูแผนที่ได้ที่: [แผนที่](https://maps.app.goo.gl/fake123)"
    client = build_client(knowledge_flow(fake) + [LLMResult(text=fake, usage=usage(100))])
    data = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "ขอลิงก์ google map"}
    ).json()
    assert "goo.gl" not in data["answer"]
    assert data["answer"] == "รีสอร์ทอยู่สะเดาค่ะ ดูแผนที่ได้ที่"
    assert "ห้ามสร้างลิงก์" in client.fake_llm.calls[2]["messages"][-1]["content"]


def test_ข้อความแม่แบบที่ไม่ได้แทนค่าต้องถูกกระทุ้ง():
    client = build_client(
        knowledge_flow("อัปโหลดสลิปภายใน {{holdMinutes}} นาทีค่ะ")
        + [LLMResult(text="อัปโหลดสลิปภายใน 3 นาทีค่ะ", usage=usage(100))]
    )
    data = client.post(
        "/api/v1/chatbot/query", json={"session_id": "s1", "message": "อัปโหลดสลิปไม่ทันเป็นไง"}
    ).json()
    assert data["answer"] == "อัปโหลดสลิปภายใน 3 นาทีค่ะ"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("มีห้องว่างค่ะ หากต้องการจองเพิ่มเติม แจ้งได้เลยค่ะ",
         "มีห้องว่างค่ะ หากต้องการจอง สามารถจองได้ในแอปเลยค่ะ"),
        ("ห้อง F1 ไม่ว่างค่ะ ถ้าต้องการจองเพิ่มเติม แจ้งได้เลยค่ะ",
         "ห้อง F1 ไม่ว่างค่ะ หากต้องการจอง สามารถจองได้ในแอปเลยค่ะ"),
        ("มีหลายห้องค่ะ\nหากต้องการจอง โปรดแจ้งชื่อห้องที่ต้องการ และจำนวนผู้เข้าพักค่ะ",
         "มีหลายห้องค่ะ\nหากต้องการจอง สามารถจองได้ในแอปเลยค่ะ"),
        ("Rooms are free. Please let me know if you'd like to proceed with a reservation.",
         "Rooms are free. You can book directly in the app."),
        ("Rooms are free.\n\nWould you like to proceed with a reservation?",
         "Rooms are free.\n\nYou can book directly in the app."),
    ],
)
def test_ไม่ชวนให้แจ้งจองในแชท_เพราะบอทจองให้ไม่ได้(answer, expected):
    from app.services.rag_pipeline import _rewrite_booking_cta

    assert _rewrite_booking_cta(answer) == expected


def test_โหลดโมเดล_embedding_ครั้งเดียวแม้มีหลาย_request_พร้อมกัน(monkeypatch):
    """เคยโหลดซ้อนกันหลาย thread ตอน container เพิ่งเปิด แล้วพัง 500 ทุก request"""
    import threading
    import time

    from app.services import embedding_service

    loads = []

    class SlowModel:
        def __init__(self):
            loads.append(threading.get_ident())
            time.sleep(0.05)

    monkeypatch.setattr(embedding_service, "EmbeddingService", SlowModel)
    monkeypatch.setattr(embedding_service, "_instance", None)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(embedding_service.get_embedding_service()))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(loads) == 1
    assert len({id(r) for r in results}) == 1


@pytest.mark.asyncio
async def test_จำนวนห้องที่ลูกค้าพิมพ์ต้องถูกส่งเข้า_quote_price():
    """'2 ห้อง 3 คืน' เคยได้ราคาห้องเดียวเพราะโมเดลไม่ส่ง rooms มา"""
    import json
    from unittest.mock import patch

    from app.core.tools import ToolExecutor
    from app.services.booking_api import BookingAPIClient

    executor = ToolExecutor(
        BookingAPIClient(), MagicMock(), MagicMock(),
        today="2026-10-08", user_message="2 ห้อง 3 คืน 25-28 ต.ค. ราคารวมเท่าไหร่",
    )
    with patch.object(BookingAPIClient, "search_rooms", return_value=FIVE_ROOMS):
        result = json.loads(await executor.run(
            "quote_price", {"check_in": "2026-10-25", "check_out": "2026-10-28"}
        ))
    assert result["rooms"] == 2
    assert result["total"] == 650 * 2 * 3


@pytest.mark.asyncio
async def test_guests_ที่โมเดลแต่งเองต้องถูกทิ้ง():
    """ทดสอบจริง '2 ห้อง 3 คืน' โมเดลส่ง guests=2 มาเอง แล้วได้ราคาห้องเดียว"""
    import json
    from unittest.mock import patch

    from app.core.tools import ToolExecutor
    from app.services.booking_api import BookingAPIClient

    executor = ToolExecutor(
        BookingAPIClient(), MagicMock(), MagicMock(),
        today="2026-10-08", user_message="2 ห้อง 3 คืน 25-28 ต.ค. ราคารวมเท่าไหร่",
        guests_hint=None,
    )
    with patch.object(BookingAPIClient, "search_rooms", return_value=FIVE_ROOMS):
        result = json.loads(await executor.run(
            "quote_price",
            {"check_in": "2026-10-25", "check_out": "2026-10-28", "rooms": 2, "guests": 2},
        ))
    assert result["total"] == 3900
    assert executor.planned_guests == []


@pytest.mark.asyncio
async def test_เพิ่มคนในการจองที่มีอยู่_ห้ามคิดเป็นห้องใหม่():
    """'การจองที่มีอยู่ ขอเพิ่มเป็น 3 คน' เคยได้ห้อง P1 มาแทนห้องเดิม F1 สองรอบติด"""
    import json

    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    executor = ToolExecutor(
        booking_client, MagicMock(), MagicMock(), customer_token="jwt", today="2026-10-08",
        user_message="การจองที่มีอยู่ ขอเพิ่มเป็น 3 คนได้ไหม", guests_hint=3,
    )
    result = json.loads(await executor.run(
        "quote_price", {"check_in": "2026-10-12", "check_out": "2026-10-18", "guests": 3}
    ))
    assert result["error"] == "existing_booking"
    booking_client.search_rooms.assert_not_awaited()


@pytest.mark.parametrize(
    ("text", "expected"),
    [("มา 4 คน", 4), ("ผู้ใหญ่ 2 เด็ก 3", 5), ("เรามาสามคน", 3), ("We are 4 people", 4),
     ("2 ห้อง 3 คืน", None)],
)
def test_นับจำนวนคนจากข้อความลูกค้า(text, expected):
    from app.core.tools import guests_mentioned

    assert guests_mentioned(text) == expected


@pytest.mark.parametrize(
    ("message", "check_out", "error"),
    [
        # เคยคิดเป็น 2 คืน
        ("อยู่ 2 อาทิตย์ ราคาเท่าไหร่", "2026-10-10", "nights_mismatch"),
        ("อยู่ 2 อาทิตย์ ราคาเท่าไหร่", "2026-10-22", None),
        ("พักสามคืน", "2026-10-09", "nights_mismatch"),
        ("2 ห้อง 3 คืน", "2026-10-11", None),
        # "3 วัน" กำกวม ปล่อยให้โมเดลเลือกแล้วบอกลูกค้าตามกติกาข้อ 7
        ("มา 3 วัน", "2026-10-10", None),
    ],
)
@pytest.mark.asyncio
async def test_จำนวนคืนต้องตรงกับที่ลูกค้าบอก(message, check_out, error):
    import json

    from app.core.tools import ToolExecutor

    booking_client = AsyncMock()
    booking_client.search_rooms.return_value = SAMPLE_ROOMS
    executor = ToolExecutor(
        booking_client, MagicMock(), MagicMock(), today="2026-10-08", user_message=message
    )
    result = json.loads(await executor.run(
        "search_available_rooms", {"check_in": "2026-10-08", "check_out": check_out}
    ))
    assert result.get("error") == error


def test_จับชื่อห้องที่จองไว้ผิด():
    """'ห้องนั้นมีตู้เย็นไหม' เคยตอบ 'ห้อง P1 ที่คุณจองไว้มีตู้เย็น' ทั้งที่จอง F1"""
    from app.services.rag_pipeline import _wrong_booked_rooms

    assert _wrong_booked_rooms("ห้อง P1 ที่คุณจองไว้มีตู้เย็นขนาดเล็กค่ะ", {"F1"}) == ["P1"]
    assert _wrong_booked_rooms("ห้องที่จองไว้คือห้อง P1 ค่ะ", {"F1"}) == ["P1"]
    assert _wrong_booked_rooms("ห้อง F1 ที่คุณจองไว้ไม่มีตู้เย็นค่ะ", {"F1"}) == []
    # ยังไม่ได้ดูการจองในรอบนี้ ตรวจไม่ได้
    assert _wrong_booked_rooms("ห้อง P1 ที่คุณจองไว้", set()) == []


def test_จับชื่อ_tool_และ_field_ที่หลุดถึงลูกค้า():
    from app.services.rag_pipeline import _leaked_internal_names

    assert _leaked_internal_names(
        "ค่าเตียงเสริมจาก search_knowledge_base คือ 200 บาท", "th"
    ) == ["search_knowledge_base"]
    assert _leaked_internal_names("ไม่มีห้องใดระบุไว้ในรายการ amenities ค่ะ", "th") == ["amenities"]
    assert _leaked_internal_names("ยังไม่ได้ชำระ (paymentStatus = unpaid)", "th") == ["paymentStatus"]
    # คำอังกฤษธรรมดาในคำตอบภาษาอังกฤษไม่ใช่การหลุด
    assert _leaked_internal_names("Amenities: Wi-Fi, TV. Maximum capacity is 2.", "en") == []


@pytest.mark.parametrize(
    ("answer", "wrong"),
    [
        # 400 คือยอดรวม 2 คืน แต่เขียนเป็นอัตราต่อคืน
        ("ค่าเตียงเสริมจากคลังความรู้คือ 400 บาทต่อเตียงต่อคืนค่ะ", [400]),
        ("เตียงเสริมคืนละ 300 บาทค่ะ", [300]),
        ("เตียงเสริมคืนละ 200 บาทต่อเตียงค่ะ", []),
        # ยอดรวมค่าเตียงเสริมไม่ใช่อัตราต่อคืน ไม่ต้องจับ
        ("ค่าเตียงเสริม 400 บาท ชำระที่รีสอร์ทในวันเช็คอิน", []),
        # ค่าห้องคืนละ 650 ที่อยู่ก่อนคำว่าเตียงเสริมต้องไม่ถูกจับ
        ("ห้อง F1 คืนละ 650 บาท และเตียงเสริมคืนละ 200 บาท", []),
        ("The extra bed is 250 THB per night.", [250]),
    ],
)
def test_จับอัตราค่าเตียงเสริมที่ไม่ตรงกับคลังความรู้(answer, wrong):
    from app.services.rag_pipeline import _wrong_extra_bed_rates

    assert _wrong_extra_bed_rates(answer, 200) == wrong


def test_จับคำตอบที่บอกว่าห้องมีของที่ไม่มี():
    """ถาม 'ห้องที่จองไว้มีตู้เย็นไหม' เคยตอบ 'ห้อง F1 มีตู้เย็นค่ะ' ทั้งที่ F1 ไม่มี"""
    from app.services.rag_pipeline import _false_amenity_claims

    amenities = {"F1": ["Wi-Fi", "TV"], "P1": ["Wi-Fi", "TV", "Mini fridge"]}
    assert _false_amenity_claims("ห้อง F1 ที่คุณจองไว้มีตู้เย็นค่ะ", amenities) == ["ห้อง F1 ไม่มีตู้เย็น"]
    assert _false_amenity_claims("ห้อง F1 ไม่มีตู้เย็นค่ะ", amenities) == []
    assert _false_amenity_claims("ห้อง P1 มีตู้เย็นขนาดเล็กค่ะ", amenities) == []
    # ประโยคที่ไล่หลายห้องไม่ตรวจ เพราะบอกไม่ได้ว่าของชิ้นไหนเป็นของห้องไหน
    assert _false_amenity_claims("ห้อง F1 และ P1 มีทีวี ส่วน P1 มีตู้เย็น", amenities) == []


@pytest.mark.asyncio
async def test_ค้นคลังความรู้ด้วยคำถามเต็มของลูกค้าด้วย():
    """โมเดลย่อคำค้นเหลือ 'สระว่ายน้ำ' แล้วหาเอกสารเรื่องสระไม่เจอ ส่วนคำถามเต็มเจออันดับ 1"""
    from app.core.tools import ToolExecutor

    vector_store = MagicMock()
    vector_store.query.side_effect = [
        [{"doc_id": "th-overview::0004", "text": "ข้อมูลทั่วไป", "distance": 0.185}],
        [{"doc_id": "th-faq::0010", "text": "ไม่มีสระว่ายน้ำ", "distance": 0.161}],
    ]
    embedding = MagicMock()
    executor = ToolExecutor(
        AsyncMock(), embedding, vector_store, today="2026-10-08", user_message="มีสระว่ายน้ำไหม"
    )
    await executor.run("search_knowledge_base", {"query": "สระว่ายน้ำ"})
    assert [call.args[0] for call in embedding.embed_query.call_args_list] == [
        "สระว่ายน้ำ", "มีสระว่ายน้ำไหม"
    ]
    assert executor.knowledge_hits[0]["doc_id"] == "th-faq::0010"


@pytest.mark.asyncio
async def test_การจองที่ยกเลิกแล้วไม่นับเป็นค้างจ่าย():
    """'ยังมีอันไหนที่ยังไม่ได้จ่าย' เคยยกรายการที่ยกเลิกไปแล้วมาบอกว่าค้างจ่าย"""
    import json

    from app.core.tools import ToolExecutor
    from app.services.booking_api import BookingAPIError

    booking_client = AsyncMock()
    booking_client.get_my_bookings.return_value = [
        _booking("F1", "2026-10-12", "2026-10-18", "approved", "paid"),
        _booking("F1", "2027-01-15", "2027-01-17", "cancelled", "unpaid"),
        _booking("P2", "2026-11-20", "2026-11-22", "pending", "unpaid"),
    ]
    booking_client.get_room.side_effect = BookingAPIError("ไม่พบ")
    executor = ToolExecutor(
        booking_client, MagicMock(), MagicMock(), customer_token="jwt", today="2026-10-08"
    )
    result = json.loads(await executor.run("get_my_bookings", {}))
    assert result["unpaidUpcomingBookings"] == ["P2-2026-11-20"]


@pytest.mark.asyncio
async def test_จำนวนคนจากข้อความก่อนหน้าต้องถูกส่งเข้า_quote_price():
    """'มา 4 คน' -> 'ห้องเดียวแล้วขอเตียงเสริมได้ไหม' เคยตอบว่าได้ โดยไม่คิดจาก 4 คน"""
    store = temp_history_store()
    store.append("s1", "มา 4 คน", "ทราบแล้วค่ะ")
    pipeline, fake_llm, _, _ = build_pipeline(
        [
            LLMResult(text="", usage=usage(100), tool_calls=[
                ToolCall(id="c1", name="search_knowledge_base", arguments={"query": "เตียงเสริม"})
            ]),
            LLMResult(text="พักห้องเดียวได้ค่ะ ขอเตียงเสริมได้", usage=usage(100)),
            LLMResult(text="ห้องเดียวไม่พอค่ะ ต้องจอง 2 ห้อง", usage=usage(100)),
        ],
        history_store=store,
    )
    response = await pipeline.answer("s1", "ห้องเดียวแล้วขอเตียงเสริมได้ไหม")
    assert response.answer == "ห้องเดียวไม่พอค่ะ ต้องจอง 2 ห้อง"
    assert "guests = 4" in fake_llm.calls[2]["messages"][-1]["content"]
