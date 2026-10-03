"""
Interface กลางสำหรับเรียก LLM (LLMClient) พร้อม implementation 2 ตัว:
- TyphoonLLMClient: เรียก Typhoon API (opentyphoon.ai) ผ่าน HTTP (OpenAI-compatible schema)
- GeminiLLMClient: เรียก Gemini API (Google AI Studio)

เวลาจะสลับ provider ให้เปลี่ยนแค่ `LLM_PROVIDER` ใน .env โค้ดส่วนอื่น (rag_pipeline) ไม่ต้องแก้เลย
เพราะเรียกผ่าน get_llm_client() ที่คืน object ตาม interface เดียวกันเสมอ

**ทำไม generate รับ messages ทั้งก้อน ไม่ใช่ prompt + system**
ตั้งแต่บอทดึงข้อมูลสดจากระบบจองได้ การตอบหนึ่งคำถามไม่ใช่การเรียก LLM ครั้งเดียวอีกต่อไป
แต่เป็นลูป: โมเดลขอเรียก tool -> เรารันแล้วยัดผลกลับเข้าไป -> โมเดลตอบ
ผลของ tool ต้องอยู่ในประวัติบทสนทนาในรูป role "tool" ที่ผูกกับ tool_call_id เดิม
ไม่งั้นโมเดลไม่รู้ว่าผลนั้นเป็นคำตอบของคำขอไหน การส่ง prompt เดี่ยว ๆ จึงไม่พอ

ยืนยันกับ API จริงแล้วว่า typhoon-v2.5-30b-a3b-instruct รองรับ tool calling ครบทั้งสองขา
(ขอเรียก tool ได้ และรับผลกลับไปสรุปเป็นคำตอบได้)
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx

from app.config import settings


@dataclass
class ToolCall:
    """คำขอเรียก tool หนึ่งครั้งจากโมเดล"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResult:
    text: str
    #: {"prompt_tokens", "completion_tokens", "total_tokens"} — None ถ้า provider ไม่คืนค่ามาให้
    usage: dict[str, int] | None = None
    #: ว่างเปล่าเมื่อโมเดลตอบเป็นข้อความเลย ไม่ได้ขอเรียก tool
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(ABC):
    """Interface กลางที่ทุก provider ต้อง implement"""

    @abstractmethod
    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResult:
        """ส่งประวัติบทสนทนาไปยัง LLM แล้วคืนคำตอบหรือคำขอเรียก tool

        Args:
            messages: ประวัติบทสนทนาแบบ OpenAI schema โดย message แรกเป็น role
                "system" เสมอ สำคัญมาก: กติกาต้องอยู่ใน role "system" ห้ามยัดรวมกับ
                คำถามใน role "user" ไม่งั้นโมเดลจะไม่ทำตามกติกา
                (ดูเหตุผลและผลทดสอบใน app/core/prompts.py)
            tools: tool schema ที่ยอมให้โมเดลเรียก None หรือ [] = ห้ามเรียก tool
        """
        raise NotImplementedError


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """แปลง arguments ที่โมเดลส่งมาเป็น dict

    โมเดลส่ง arguments มาเป็น JSON string ที่มันเขียนเอง จึงพังได้
    คืน dict ว่างเมื่อ parse ไม่ได้ แล้วปล่อยให้ ToolExecutor รายงาน bad_arguments
    กลับไปให้โมเดลแก้ตัวในรอบถัดไป ดีกว่าให้ทั้ง request พัง
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


#: บางรอบ Typhoon ไม่ได้ใช้ field tool_calls แต่พิมพ์คำขอเรียก tool ออกมาเป็นข้อความธรรมดา
#: ในรูป <tool_call>{"name": ..., "arguments": {...}}</tool_call> ซึ่งถ้าปล่อยไว้
#: ลูกค้าจะเห็น JSON ดิบเป็นคำตอบ (เจอจริงกับคำถาม "เรามา 3 วัน นอนเตียงเดี่ยว ราคาเท่าไหร่")
#: จึงแปลงกลับเป็น ToolCall ให้เหมือนกับที่มันควรจะส่งมาตั้งแต่แรก
_INLINE_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_inline_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """ดึงคำขอเรียก tool ที่ปนมาในเนื้อข้อความออกมา พร้อมคืนข้อความที่ตัดบล็อกนั้นออกแล้ว"""
    calls: list[ToolCall] = []
    for index, match in enumerate(_INLINE_TOOL_CALL.finditer(text)):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        if not name:
            continue
        arguments = payload.get("arguments", payload.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = _parse_arguments(arguments)
        calls.append(
            ToolCall(
                id=f"inline-{index}-{name}",
                name=name,
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return _INLINE_TOOL_CALL.sub("", text).strip(), calls


class TyphoonLLMClient(LLMClient):
    """เรียก Typhoon API ซึ่งใช้ schema แบบเดียวกับ OpenAI chat completion"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.typhoon_api_key
        self.base_url = base_url or settings.typhoon_base_url
        self.model = model or settings.typhoon_model

    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            data = response.json()

        message = data["choices"][0]["message"]
        usage = data.get("usage")
        # content เป็น null เมื่อโมเดลขอเรียก tool แทนที่จะตอบเป็นข้อความ
        text = message.get("content") or ""
        tool_calls = [
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=_parse_arguments(call["function"].get("arguments")),
            )
            for call in (message.get("tool_calls") or [])
        ]
        if not tool_calls:
            text, tool_calls = _extract_inline_tool_calls(text)

        return LLMResult(
            text=text,
            usage=(
                {
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                }
                if usage
                else None
            ),
            tool_calls=tool_calls,
        )


class GeminiLLMClient(LLMClient):
    """เรียก Gemini API ผ่าน Google AI Studio (generativelanguage.googleapis.com)

    Gemini ใช้ชื่อ field คนละชุดกับ OpenAI ทั้งหมด ทั้งโครง message, การประกาศ tool
    และการส่งผล tool กลับ จึงต้องแปลงไปมาในคลาสนี้ เพื่อให้ rag_pipeline
    ทำงานกับ schema เดียวไม่ว่าจะตั้ง provider เป็นอะไร
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    @staticmethod
    def _to_gemini_contents(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """แปลง messages แบบ OpenAI เป็น contents + systemInstruction ของ Gemini"""
        contents: list[dict[str, Any]] = []
        system_instruction: dict[str, Any] | None = None

        for message in messages:
            role = message.get("role")
            if role == "system":
                system_instruction = {"parts": [{"text": message["content"]}]}
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": message["content"]}]})
            elif role == "assistant":
                if message.get("tool_calls"):
                    contents.append(
                        {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": call["function"]["name"],
                                        "args": _parse_arguments(
                                            call["function"].get("arguments")
                                        ),
                                    }
                                }
                                for call in message["tool_calls"]
                            ],
                        }
                    )
                else:
                    contents.append(
                        {"role": "model", "parts": [{"text": message.get("content") or ""}]}
                    )
            elif role == "tool":
                # Gemini ไม่มี role "tool" — ผลของ tool ถูกส่งกลับในฝั่ง user
                # และผูกกับ tool ด้วย "name" ไม่ใช่ tool_call_id แบบ OpenAI
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.get("name", ""),
                                    "response": {"result": message["content"]},
                                }
                            }
                        ],
                    }
                )
        return contents, system_instruction

    @staticmethod
    def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations = []
        for tool in tools:
            function = dict(tool["function"])
            # Gemini ไม่ยอมรับ parameters ที่ไม่มี property เลย ต้องตัดทิ้งไปทั้ง key
            if not function.get("parameters", {}).get("properties"):
                function.pop("parameters", None)
            declarations.append(function)
        return [{"functionDeclarations": declarations}]

    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> LLMResult:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        contents, system_instruction = self._to_gemini_contents(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if tools:
            payload["tools"] = self._to_gemini_tools(tools)

        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        parts = data["candidates"][0]["content"].get("parts", [])
        usage = data.get("usageMetadata")
        tool_calls = [
            # Gemini ไม่ได้ให้ id ของ tool call มา แต่ฝั่งเรา (และ OpenAI schema) ต้องมี
            # จึงสร้างจากลำดับ+ชื่อ ซึ่งไม่ซ้ำกันภายในหนึ่งรอบอยู่แล้ว
            ToolCall(
                id=f"gemini-{index}-{part['functionCall']['name']}",
                name=part["functionCall"]["name"],
                arguments=part["functionCall"].get("args") or {},
            )
            for index, part in enumerate(parts)
            if "functionCall" in part
        ]
        return LLMResult(
            text="".join(part.get("text", "") for part in parts),
            usage=(
                {
                    "prompt_tokens": usage["promptTokenCount"],
                    # ไม่มี candidatesTokenCount ในรอบที่โมเดลตอบด้วย functionCall ล้วน
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": usage["totalTokenCount"],
                }
                if usage
                else None
            ),
            tool_calls=tool_calls,
        )


@lru_cache
def get_llm_client() -> LLMClient:
    """Factory: คืนค่า LLMClient ตาม provider ที่ตั้งค่าไว้ใน settings.llm_provider"""
    if settings.llm_provider == "gemini":
        return GeminiLLMClient()
    return TyphoonLLMClient()
