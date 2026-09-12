"""
Interface กลางสำหรับเรียก LLM (LLMClient) พร้อม implementation 2 ตัว:
- TyphoonLLMClient: เรียก Typhoon API (opentyphoon.ai) ผ่าน HTTP (OpenAI-compatible schema)
- GeminiLLMClient: เรียก Gemini API (Google AI Studio)

เวลาจะสลับ provider ให้เปลี่ยนแค่ `LLM_PROVIDER` ใน .env โค้ดส่วนอื่น (rag_pipeline) ไม่ต้องแก้เลย
เพราะเรียกผ่าน get_llm_client() ที่คืน object ตาม interface เดียวกันเสมอ
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import httpx

from app.config import settings


@dataclass
class LLMResult:
    text: str
    #: {"prompt_tokens", "completion_tokens", "total_tokens"} — None ถ้า provider ไม่คืนค่ามาให้
    usage: dict[str, int] | None = None


class LLMClient(ABC):
    """Interface กลางที่ทุก provider ต้อง implement"""

    @abstractmethod
    async def generate(self, prompt: str, system: str | None = None) -> LLMResult:
        """ส่ง prompt ไปยัง LLM แล้วคืนคำตอบพร้อมจำนวน token ที่ใช้

        Args:
            prompt: ข้อความของผู้ใช้ (ข้อมูลอ้างอิง + คำถาม)
            system: กติกาที่โมเดลต้องทำตาม ส่งแยกเป็น system role
                สำคัญมาก: ถ้ายัดรวมไปกับ prompt โมเดลจะไม่ทำตามกติกา
                (ดูเหตุผลและผลทดสอบใน app/core/prompts.py)
        """
        raise NotImplementedError


class TyphoonLLMClient(LLMClient):
    """เรียก Typhoon API ซึ่งใช้ schema แบบเดียวกับ OpenAI chat completion"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.typhoon_api_key
        self.base_url = base_url or settings.typhoon_base_url
        self.model = model or settings.typhoon_model

    async def generate(self, prompt: str, system: str | None = None) -> LLMResult:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage")
            return LLMResult(
                text=data["choices"][0]["message"]["content"],
                usage=(
                    {
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                    }
                    if usage
                    else None
                ),
            )


class GeminiLLMClient(LLMClient):
    """เรียก Gemini API ผ่าน Google AI Studio (generativelanguage.googleapis.com)"""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def generate(self, prompt: str, system: str | None = None) -> LLMResult:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload: dict = {"contents": [{"parts": [{"text": prompt}]}]}
        if system:
            # Gemini รับ system prompt คนละ field กับ OpenAI schema
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # ชื่อ field คนละชุดกับ OpenAI schema แต่ความหมายตรงกัน
            usage = data.get("usageMetadata")
            return LLMResult(
                text=data["candidates"][0]["content"]["parts"][0]["text"],
                usage=(
                    {
                        "prompt_tokens": usage["promptTokenCount"],
                        "completion_tokens": usage["candidatesTokenCount"],
                        "total_tokens": usage["totalTokenCount"],
                    }
                    if usage
                    else None
                ),
            )


@lru_cache
def get_llm_client() -> LLMClient:
    """Factory: คืนค่า LLMClient ตาม provider ที่ตั้งค่าไว้ใน settings.llm_provider"""
    if settings.llm_provider == "gemini":
        return GeminiLLMClient()
    return TyphoonLLMClient()
