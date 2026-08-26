"""
Interface กลางสำหรับเรียก LLM (LLMClient) พร้อม implementation 2 ตัว:
- TyphoonLLMClient: เรียก Typhoon API (opentyphoon.ai) ผ่าน HTTP (OpenAI-compatible schema)
- GeminiLLMClient: เรียก Gemini API (Google AI Studio)

เวลาจะสลับ provider ให้เปลี่ยนแค่ `LLM_PROVIDER` ใน .env โค้ดส่วนอื่น (rag_pipeline) ไม่ต้องแก้เลย
เพราะเรียกผ่าน get_llm_client() ที่คืน object ตาม interface เดียวกันเสมอ
"""

from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from app.config import settings


class LLMClient(ABC):
    """Interface กลางที่ทุก provider ต้อง implement"""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """ส่ง prompt ไปยัง LLM แล้วคืนค่าคำตอบเป็นข้อความ"""
        raise NotImplementedError


class TyphoonLLMClient(LLMClient):
    """เรียก Typhoon API ซึ่งใช้ schema แบบเดียวกับ OpenAI chat completion"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.typhoon_api_key
        self.base_url = base_url or settings.typhoon_base_url
        self.model = model or settings.typhoon_model

    async def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


class GeminiLLMClient(LLMClient):
    """เรียก Gemini API ผ่าน Google AI Studio (generativelanguage.googleapis.com)"""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    async def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]


@lru_cache
def get_llm_client() -> LLMClient:
    """Factory: คืนค่า LLMClient ตาม provider ที่ตั้งค่าไว้ใน settings.llm_provider"""
    if settings.llm_provider == "gemini":
        return GeminiLLMClient()
    return TyphoonLLMClient()
