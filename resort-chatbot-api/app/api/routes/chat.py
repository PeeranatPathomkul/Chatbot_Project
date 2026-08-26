"""Endpoint สำหรับ chatbot: query และ health check"""

from fastapi import APIRouter, Depends

from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.rag_pipeline import RAGPipeline, get_rag_pipeline

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """ตรวจสอบว่า service ยังทำงานปกติ"""
    return HealthResponse(status="ok")


@router.post("/query", response_model=ChatResponse)
async def query(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ChatResponse:
    """รับคำถามจากผู้ใช้ แล้วตอบกลับด้วย RAG pipeline"""
    return await pipeline.answer(
        session_id=request.session_id,
        resort_id=request.resort_id,
        message=request.message,
        language=request.language,
    )
