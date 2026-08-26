"""Pydantic models สำหรับ request/response ของ endpoint chatbot"""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="รหัสของ session สนทนา ใช้แยกบทสนทนาของผู้ใช้แต่ละคน")
    resort_id: str = Field(..., description="รหัสของรีสอร์ตที่ผู้ใช้กำลังคุยด้วย")
    message: str = Field(..., description="ข้อความคำถามจากผู้ใช้")
    language: str = Field(default="th", description="ภาษาที่ต้องการให้ตอบ เช่น th, en")


class Source(BaseModel):
    doc_id: str = Field(..., description="รหัสเอกสารต้นทางที่ใช้อ้างอิงคำตอบ")
    snippet: str = Field(..., description="ข้อความบางส่วนจากเอกสารที่นำมาใช้ตอบ")


class SuggestedAction(BaseModel):
    type: Literal["link", "none"] = "none"
    url: str = ""


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_action: SuggestedAction = Field(default_factory=SuggestedAction)


class HealthResponse(BaseModel):
    status: str = "ok"
