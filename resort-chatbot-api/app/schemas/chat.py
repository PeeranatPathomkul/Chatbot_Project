"""Pydantic models สำหรับ request/response ของ endpoint chatbot"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="รหัสของ session สนทนา ใช้แยกบทสนทนาของผู้ใช้แต่ละคน")
    message: str = Field(..., description="ข้อความคำถามจากผู้ใช้")
    language: str = Field(default="th", description="ภาษาที่ต้องการให้ตอบ: th หรือ en")
    resort_id: str | None = Field(
        default=None,
        description=(
            "ไม่ได้ใช้งานแล้ว — คลังความรู้ปัจจุบันเป็นรีสอร์ทเดียว "
            "คงไว้เพื่อไม่ให้ client เดิมพัง ค่าที่ส่งมาจะถูกเพิกเฉย"
        ),
    )


class Source(BaseModel):
    doc_id: str = Field(..., description="รหัสเอกสารต้นทางที่ใช้อ้างอิงคำตอบ")
    snippet: str = Field(..., description="ข้อความบางส่วนจากเอกสารที่นำมาใช้ตอบ")


class SuggestedAction(BaseModel):
    type: Literal["link", "none"] = "none"
    url: str = ""


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source] = Field(
        default_factory=list,
        description="เอกสารที่ใช้ตอบ — ว่างเปล่าเมื่อ answered=false เพราะไม่ได้ใช้ตอบจริง",
    )
    answered: bool = Field(
        default=True,
        description=(
            "true = ตอบคำถามได้จากข้อมูลในระบบ, "
            "false = ไม่มีข้อมูลจึงแนะนำให้ติดต่อรีสอร์ทโดยตรง "
            "ใช้ field นี้ตัดสินใจว่าจะแสดงปุ่มโทรหรือไม่"
        ),
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "ความมั่นใจของคำตอบ (0.0 เมื่อ answered=false) "
            "เตือน: ค่านี้ไม่ใช่เครื่องตรวจจับการแต่งข้อมูล ให้ดู answered เป็นหลัก"
        ),
    )
    retrieval_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "cosine similarity ของ chunk อันดับ 1 ดิบ ๆ ไว้ใช้ debug และ monitor "
            "อย่านำไปตั้ง threshold ตัดคำตอบ เพราะคำถามที่ระบบไม่มีคำตอบ "
            "ก็ได้ค่านี้สูงราว 0.83 พอ ๆ กับคำถามที่ตอบได้"
        ),
    )
    suggested_action: SuggestedAction = Field(default_factory=SuggestedAction)


class HealthResponse(BaseModel):
    status: str = "ok"
