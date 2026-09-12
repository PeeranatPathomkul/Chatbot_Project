"""
Test สำหรับ endpoint /api/v1/chatbot/query และ /api/v1/chatbot/health

หมายเหตุ: mock LLMClient และ VectorStore/EmbeddingService ไว้ทั้งหมด
เพื่อไม่ให้ยิง API จริงหรือโหลดโมเดล embedding จริงตอนรัน test (ประหยัดเวลา/เงิน)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.prompts import NO_CONTEXT_ANSWER
from app.main import app
from app.services.embedding_service import get_embedding_service
from app.services.llm_client import LLMResult, get_llm_client
from app.services.rag_pipeline import RAGPipeline, get_rag_pipeline
from app.services.vector_store import get_vector_store

#: chunk ตัวอย่างในรูปแบบที่ indexing-pipeline สร้างจริง — มีบรรทัดบริบทนำหน้า
SAMPLE_CHUNK_TEXT = (
    "[หมวด: นโยบาย | หัวข้อ: เวลาเช็คอินและเช็คเอาท์]\n"
    "เวลาเช็คอินคือ 14.00 น. เป็นต้นไป\n"
    "รับเช็คอินได้ถึงเวลา 22.00 น."
)


def build_client(llm_answer: str) -> TestClient:
    """สร้าง TestClient ที่ mock ทุก dependency โดยกำหนดคำตอบของ LLM ได้"""
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = [
        {
            "doc_id": "th-booking_policy::0000",
            "text": SAMPLE_CHUNK_TEXT,
            "metadata": {"source": "th/booking_policy.md", "language": "th"},
            "distance": 0.12,
        }
    ]

    mock_llm_client = AsyncMock()
    mock_llm_client.generate.return_value = LLMResult(
        text=llm_answer,
        usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    )

    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client
    app.dependency_overrides[get_rag_pipeline] = lambda: RAGPipeline(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
    )

    client = TestClient(app)
    client.mock_vector_store = mock_vector_store  # type: ignore[attr-defined]
    client.mock_llm_client = mock_llm_client  # type: ignore[attr-defined]
    return client


@pytest.fixture
def client():
    test_client = build_client("เช็คอินได้ตั้งแต่เวลา 14.00 น. ถึง 22.00 น. ค่ะ")
    with test_client as c:
        c.mock_vector_store = test_client.mock_vector_store  # type: ignore[attr-defined]
        c.mock_llm_client = test_client.mock_llm_client  # type: ignore[attr-defined]
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def refusing_client():
    """client ที่ LLM ปฏิเสธเพราะไม่มีข้อมูลในคลัง"""
    test_client = build_client(NO_CONTEXT_ANSWER)
    with test_client as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client: TestClient):
    response = client.get("/api/v1/chatbot/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_from_mocked_llm(client: TestClient):
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
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["token_usage"] == {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }


def test_ไม่ต้องส่ง_resort_id_ก็เรียกได้(client: TestClient):
    """resort_id เป็นของเหลือจากตอนออกแบบ multi-tenant ไม่ควรบังคับอีกต่อไป"""
    response = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "เช็คอินได้กี่โมง"},
    )
    assert response.status_code == 200


def test_ส่ง_resort_id_มาก็ยังรับได้_แต่ไม่พัง(client: TestClient):
    """client เดิมที่ยังส่ง resort_id มาต้องไม่ error"""
    response = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "resort_id": "sunrise_villa", "message": "เช็คอินได้กี่โมง"},
    )
    assert response.status_code == 200


def test_กรอง_chunk_ตามภาษาที่ลูกค้าถาม(client: TestClient):
    """ถ้าไม่กรองภาษา ลูกค้าที่ถามไทยอาจได้ chunk อังกฤษกลับไป"""
    client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "What time is check-in?", "language": "en"},
    )
    assert client.mock_vector_store.query.call_args.kwargs["language"] == "en"


def test_ภาษาที่ไม่รองรับถูกแทนด้วยค่าเริ่มต้น(client: TestClient):
    client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "test", "language": "ja"},
    )
    assert client.mock_vector_store.query.call_args.kwargs["language"] == "th"


def test_กติกาถูกส่งแยกเป็น_system_ไม่ใช่ยัดรวมกับคำถาม(client: TestClient):
    """จุดนี้คือสิ่งที่ทำให้โมเดลยอมปฏิเสธแทนที่จะแต่งคำตอบ ถ้าหลุดต้องรู้ทันที"""
    client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "เช็คอินได้กี่โมง"},
    )
    kwargs = client.mock_llm_client.generate.call_args.kwargs
    assert "ห้ามใช้ความรู้ทั่วไปของคุณเอง" in kwargs["system"]

    user_prompt = client.mock_llm_client.generate.call_args.args[0]
    assert "ห้ามใช้ความรู้ทั่วไปของคุณเอง" not in user_prompt
    assert "เช็คอินได้กี่โมง" in user_prompt


def test_snippet_ต้องไม่มีบรรทัดบริบทติดไปให้ลูกค้าเห็น(client: TestClient):
    response = client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "เช็คอินได้กี่โมง"},
    )
    snippet = response.json()["sources"][0]["snippet"]
    assert not snippet.startswith("[หมวด:")
    assert "เวลาเช็คอินคือ 14.00 น." in snippet


def test_เมื่อปฏิเสธ_confidence_ต้องเป็นศูนย์และไม่อ้างแหล่งที่มา(refusing_client: TestClient):
    """confidence 0.84 คู่กับคำตอบ 'ไม่มีข้อมูล' จะทำให้ frontend ตัดสินใจผิด"""
    response = refusing_client.post(
        "/api/v1/chatbot/query",
        json={"session_id": "s1", "message": "จากสนามบินหาดใหญ่มากี่กิโล"},
    )

    data = response.json()
    assert data["answered"] is False
    assert data["confidence"] == 0.0
    assert data["sources"] == []
    # แต่คะแนนดิบต้องยังเก็บไว้ให้ debug ได้ว่า retrieval ดึงอะไรมา
    assert data["retrieval_score"] > 0.0
    # token ถูกใช้จริงแม้โมเดลจะปฏิเสธ (มันยังคง generate คำตอบปฏิเสธออกมา) จึงต้องรายงานด้วย
    assert data["token_usage"]["total_tokens"] > 0
