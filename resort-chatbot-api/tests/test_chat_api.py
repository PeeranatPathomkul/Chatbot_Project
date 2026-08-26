"""
Test เบื้องต้นสำหรับ endpoint /api/v1/chatbot/query และ /api/v1/chatbot/health

หมายเหตุ: mock LLMClient และ VectorStore/EmbeddingService ไว้ทั้งหมด
เพื่อไม่ให้ยิง API จริงหรือโหลดโมเดล embedding จริงตอนรัน test (ประหยัดเวลา/เงิน)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.embedding_service import get_embedding_service
from app.services.llm_client import get_llm_client
from app.services.rag_pipeline import RAGPipeline, get_rag_pipeline
from app.services.vector_store import get_vector_store


@pytest.fixture
def mock_pipeline_dependencies():
    """เตรียม mock ของ embedding service, vector store และ llm client"""
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]

    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = [
        {
            "doc_id": "row0-chunk0",
            "text": "คำถาม: เช็คอินได้กี่โมง\nคำตอบ: เช็คอินได้ตั้งแต่เวลา 14:00 น.",
            "metadata": {"resort_id": "sunrise_villa"},
            "distance": 0.1,
        }
    ]

    mock_llm_client = AsyncMock()
    mock_llm_client.generate.return_value = "เช็คอินได้ตั้งแต่เวลา 14:00 น. ค่ะ"

    return mock_embedding_service, mock_vector_store, mock_llm_client


@pytest.fixture
def client(mock_pipeline_dependencies):
    mock_embedding_service, mock_vector_store, mock_llm_client = mock_pipeline_dependencies

    app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_llm_client] = lambda: mock_llm_client
    app.dependency_overrides[get_rag_pipeline] = lambda: RAGPipeline(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_client=mock_llm_client,
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_health(client: TestClient):
    response = client.get("/api/v1/chatbot/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_from_mocked_llm(client: TestClient):
    payload = {
        "session_id": "session-123",
        "resort_id": "sunrise_villa",
        "message": "เช็คอินได้กี่โมง",
        "language": "th",
    }

    response = client.post("/api/v1/chatbot/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-123"
    assert "14:00" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["doc_id"] == "row0-chunk0"
    assert 0.0 <= data["confidence"] <= 1.0
