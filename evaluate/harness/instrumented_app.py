"""
แอปตัวจริง (app.main:app จาก working tree) ที่ติดตัวบันทึกการทำงานภายในไว้ สำหรับรันประเมินเท่านั้น

**ทำไมต้องมีไฟล์นี้** — response ของ /chatbot/query คืนแค่ชื่อ tool (tools_used) ไม่คืน arguments
แต่การวัด Argument Accuracy ต้องรู้ว่าโมเดลส่งวันที่ / จำนวนคน / จำนวนห้องอะไรมา และ Faithfulness ต้องรู้
ผลของ tool เต็ม ๆ ที่โมเดลเห็น (sources.snippet ยาวแค่ 200 ตัวอักษร) ทางที่แก้โค้ดน้อยที่สุดคือ
ไม่แก้เลย: ไฟล์นี้ import แอปจริงแล้วห่อ method สองตัวด้วย monkeypatch ที่ **แค่บันทึก ไม่เปลี่ยนผล**
- ``ToolExecutor.run``   -> ชื่อ tool, arguments, ผลที่คืนให้โมเดล, เวลา
- ``*LLMClient.generate`` -> เวลา, token, จำนวน message ที่ส่ง (ใช้ยืนยันว่ามีประวัติแชทจริง), tool ที่ขอ

ผลของแต่ละ request ถูกเก็บในหน่วยความจำ แล้ว harness ดึงทาง ``GET /_eval/trace/{id}``
(id อยู่ใน header ``X-Eval-Trace-Id`` ของ response) — response body ของแอปไม่ถูกแตะเลย

ตั้ง CHAT_HISTORY_PATH ไปที่ evaluate/.state/ ก่อน import แอป ประวัติแชทของรอบประเมิน
จึงไม่ปนกับ resort-chatbot-api/data/chat_history.db ที่ใช้พัฒนา

วิธีรัน (จากโฟลเดอร์ evaluate/)::

    ..\\resort-chatbot-api\\.venv\\Scripts\\python.exe -m uvicorn harness.instrumented_app:app --port 8000
"""

from __future__ import annotations

import contextvars
import copy
import os
import time
import uuid
from collections import OrderedDict
from typing import Any

from harness.config import STATE_DIR, add_api_to_path

STATE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CHAT_HISTORY_PATH", str(STATE_DIR / "chat_history.db"))
add_api_to_path()

from fastapi import Request  # noqa: E402

from app.config import settings  # noqa: E402
from app.core import tools as tools_module  # noqa: E402
from app.main import app  # noqa: E402
from app.services import llm_client as llm_module  # noqa: E402

_current_trace: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "eval_trace", default=None
)
_TRACES: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_TRACES = 5000
_STATS = {"requests_served": 0, "started_at": time.time()}

_original_run = tools_module.ToolExecutor.run


async def _traced_run(self, name: str, arguments: dict[str, Any]) -> str:
    started = time.perf_counter()
    output = await _original_run(self, name, arguments)
    trace = _current_trace.get()
    if trace is not None:
        trace["tool_calls"].append(
            {
                "name": name,
                "arguments": copy.deepcopy(arguments),
                "output": output,
                "ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    return output


tools_module.ToolExecutor.run = _traced_run


def _wrap_generate(cls: type) -> None:
    original = cls.generate

    async def traced_generate(self, messages, tools=None):
        started = time.perf_counter()
        result = await original(self, messages, tools=tools)
        trace = _current_trace.get()
        if trace is not None:
            trace["llm_calls"].append(
                {
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                    "usage": result.usage,
                    "messages_sent": len(messages),
                    # ประวัติ = ทุกอย่างระหว่าง system กับข้อความล่าสุดของลูกค้า ในการเรียกครั้งแรกของรอบ
                    "history_messages": max(0, len(messages) - 2) if not trace["llm_calls"] else None,
                    "tools_offered": bool(tools),
                    "tool_calls": [c.name for c in result.tool_calls],
                    "text_preview": (result.text or "")[:300],
                }
            )
        return result

    cls.generate = traced_generate


for _cls in (llm_module.TyphoonLLMClient, llm_module.GeminiLLMClient):
    _wrap_generate(_cls)


@app.middleware("http")
async def _eval_trace_middleware(request: Request, call_next):
    if not request.url.path.endswith("/chatbot/query"):
        return await call_next(request)
    # dict ตัวเดียวกันถูกแชร์เข้าไปใน task ของ endpoint (context ถูก copy ตอนสร้าง task แต่ object เดียวกัน)
    trace: dict[str, Any] = {"tool_calls": [], "llm_calls": []}
    token = _current_trace.set(trace)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        _current_trace.reset(token)
    trace["server_ms"] = round((time.perf_counter() - started) * 1000, 1)
    trace["request_index"] = _STATS["requests_served"]
    _STATS["requests_served"] += 1
    trace_id = uuid.uuid4().hex
    _TRACES[trace_id] = trace
    while len(_TRACES) > _MAX_TRACES:
        _TRACES.popitem(last=False)
    response.headers["X-Eval-Trace-Id"] = trace_id
    response.headers["X-Eval-Instrumented"] = "1"
    return response


@app.get("/_eval/trace/{trace_id}", include_in_schema=False)
async def _eval_trace(trace_id: str) -> dict[str, Any]:
    return _TRACES.get(trace_id) or {"error": "not_found"}


@app.get("/_eval/info", include_in_schema=False)
async def _eval_info() -> dict[str, Any]:
    """ค่าตั้งของแอปที่ใช้รายงาน — ไม่มีค่าลับ"""
    provider = settings.llm_provider
    return {
        "instrumented": True,
        "requests_served": _STATS["requests_served"],
        "uptime_seconds": round(time.time() - _STATS["started_at"], 1),
        "llm_provider": provider,
        "llm_model": settings.gemini_model if provider == "gemini" else settings.typhoon_model,
        "embedding_model": settings.embedding_model,
        "retrieval_top_k": settings.retrieval_top_k,
        "agent_max_tool_rounds": settings.agent_max_tool_rounds,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "chroma_collection": settings.chroma_collection_name,
        "chat_history_path": settings.chat_history_path,
        "chat_history_max_messages": settings.chat_history_max_messages,
        "booking_api_base_url": settings.booking_api_base_url,
    }
