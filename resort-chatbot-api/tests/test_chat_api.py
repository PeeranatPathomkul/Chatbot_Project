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


def test_จำนวนผู้เข้าพักสูงสุดถูกคำนวณในโค้ด_ไม่ปล่อยให้โมเดลบวกเอง():
    """โมเดลยึดเลข capacity ใน JSON มากกว่ากติกาที่เป็นร้อยแก้ว

    ทดสอบจริงแล้วคำถามเดียวกัน ("มา 3 คน พักห้องเดียวได้ไหม") ตอบไม่ตรงกันระหว่างรอบ
    การส่งค่าที่คำนวณเสร็จแล้วไปเลยตัดความไม่แน่นอนนี้ทิ้ง
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
            "amenities": [],
        }
    )
    assert trimmed["capacity"] == 2
    assert trimmed["maxGuestsWithExtraBed"] == 3


@pytest.mark.asyncio
async def test_quote_price_คิดเลขให้เสร็จในโค้ด():
    """โมเดลคิดเลขผิดเป็นระยะ เช่น 650 x 1 ห้อง x 2 คืน = 2,600 (ที่ถูกคือ 1,300)

    ผิดแบบนี้ลูกค้าจับไม่ได้เพราะสูตรที่เขียนกำกับถูก ผิดแค่ผลลัพธ์
    """
    from unittest.mock import patch

    from app.services.booking_api import BookingAPIClient

    client = BookingAPIClient()
    rooms = [
        {"id": "1", "name": "P1", "type": "single", "pricePerNight": 650, "capacity": 2,
         "typeLabel": "เตียงเดี่ยว (มีเตียงใหญ่ 1 เตียง)", "description": "", "amenities": []}
    ]
    with patch.object(BookingAPIClient, "search_rooms", return_value=rooms):
        quote = await client.quote_stay("2026-09-24", "2026-09-26")
    assert quote["nights"] == 2          # half-open: เข้า 24 ออก 26 = 2 คืน
    assert quote["roomTotal"] == 1300    # ไม่ใช่ 2,600
    assert quote["extraBedTotal"] == 0   # ไม่มีใครขอเตียงเสริม ต้องไม่โผล่มาในบิล
    assert quote["total"] == 1300

    with patch.object(BookingAPIClient, "search_rooms", return_value=rooms):
        quote = await client.quote_stay(
            "2026-09-24", "2026-09-26", extra_beds=1, extra_bed_price_per_night=200
        )
    assert quote["extraBedTotal"] == 400  # 200 x 1 เตียง x 2 คืน
    assert quote["total"] == 1700


@pytest.mark.asyncio
async def test_quote_price_ไม่ยอมคิดค่าเตียงเสริมโดยไม่รู้อัตราจริง():
    """กันไม่ให้โมเดลเดาค่าเตียงเสริมเอง ต้องไปค้นจากคลังความรู้มาก่อน"""
    from app.services.booking_api import BookingAPIClient, BookingAPIError

    with pytest.raises(BookingAPIError, match="extra_bed_price_per_night"):
        await BookingAPIClient().quote_stay("2026-09-24", "2026-09-26", extra_beds=1)


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
    """capacity ใน DB คือจำนวนคนที่นอนได้ด้วยเตียงที่มีอยู่ ไม่ใช่จำนวนสูงสุดที่ห้องรับได้

    การส่ง ?guests=3 จะกรองห้องจริงทิ้งหมด เรื่องเตียงเสริมต้องคิดในชั้นบอท
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
