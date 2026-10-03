"""Endpoint สำหรับ chatbot: query และ health check"""

from fastapi import APIRouter, Depends, Header

from app.schemas.chat import ChatRequest, ChatResponse, HealthResponse
from app.services.rag_pipeline import RAGPipeline, get_rag_pipeline

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


def _extract_bearer_token(authorization: str | None) -> str | None:
    """ดึง access token ออกจาก header Authorization

    คืน None เมื่อไม่มี header หรือรูปแบบไม่ใช่ Bearer — ถือว่าเป็นลูกค้าที่ยังไม่ล็อกอิน
    ซึ่งยังถามเรื่องห้องว่าง ราคา และนโยบายได้ตามปกติ ต่างกันแค่ถามถึงการจองของตัวเองไม่ได้
    """
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """ตรวจสอบว่า service ยังทำงานปกติ"""
    return HealthResponse(status="ok")


@router.post("/query", response_model=ChatResponse)
async def query(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    """รับคำถามจากผู้ใช้ แล้วตอบกลับด้วย RAG pipeline

    **token ของลูกค้ามาจาก header เท่านั้น ไม่ใช่ field ใน body** — ถ้าเป็น field ใน body
    โมเดลจะเห็นมันเป็นข้อมูลชิ้นหนึ่งที่แต่งขึ้นเองหรือถูกสั่งให้เปลี่ยนได้
    ตัวตนของลูกค้าต้องถูกกำหนดโดยแอปที่เรียกเรา ไม่ใช่โดยข้อความในบทสนทนา
    """
    return await pipeline.answer(
        session_id=request.session_id,
        message=request.message,
        language=request.language,
        customer_token=_extract_bearer_token(authorization),
    )
