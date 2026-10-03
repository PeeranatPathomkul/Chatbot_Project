"""
ประกอบ flow การตอบคำถามทั้งหมดเข้าด้วยกัน

**เดิมเป็น RAG รอบเดียว ตอนนี้เป็นลูป tool calling**
ของเดิม: embed คำถาม -> ค้น vector store -> ยัด context เข้า prompt -> เรียก LLM ครั้งเดียว
ซึ่งตอบได้เฉพาะสิ่งที่ถูก index ไว้ล่วงหน้า จึงตอบ "ห้องว่างไหม" ไม่ได้เลย
เพราะคำตอบเปลี่ยนทุกครั้งที่มีคนจอง และ "ราคาเท่าไหร่" ก็ตอบจากข้อความที่เน่าได้

ตอนนี้: ให้โมเดลเลือกเองว่าจะถามที่ไหน ระหว่างระบบจองจริง (PostgreSQL ผ่าน Booking API)
กับคลังความรู้ แล้ววนจนกว่าจะได้คำตอบ ชื่อคลาสยังเป็น RAGPipeline เพื่อไม่ให้ client
และ dependency override เดิมพัง แต่เนื้อในเป็นลูป agent แล้ว

**ทำไมต้องจำกัดจำนวนรอบ** — โมเดลเรียก tool ที่คืน error แล้วลองใหม่วนไปเรื่อย ๆ ได้
ครบรอบแล้วเราบังคับให้สรุปด้วยการเรียกซ้ำโดยไม่ส่ง tools ไปด้วย ซึ่งทำให้มันต้อง
ตอบเป็นข้อความเท่านั้น ดีกว่าปล่อยให้ค่า token บานหรือคืน response เปล่า
"""

import json
import re
from datetime import datetime, timedelta, timezone

from app.config import settings
from app.core.prompts import SYSTEM_PROMPT, no_context_answer
from app.core.tools import TOOL_SCHEMAS, ToolExecutor
from app.schemas.chat import ChatResponse, Source, SuggestedAction, TokenUsage
from app.services.booking_api import BookingAPIClient, get_booking_api_client
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStore, get_vector_store

#: ขึ้นต้นของบรรทัดบริบทที่ indexing-pipeline เติมไว้หัว chunk
#: (ดู chunking._build_context_header ในโปรเจกต์ indexing-pipeline)
#: บรรทัดนี้มีไว้ช่วย embedding ไม่ใช่ให้คนอ่าน จึงต้องตัดออกก่อนส่งกลับเป็น snippet
_CONTEXT_HEADER_PREFIXES = ("[หมวด:", "[Category:")

#: เวลาประเทศไทย UTC+7 — เขียนเป็น offset ตายตัวแทนการใช้ ZoneInfo
#: เพราะไทยไม่มี DST ค่านี้จึงถูกต้องตลอดปี และไม่ต้องพึ่ง tzdata ที่ Windows ไม่มีมาให้
_THAI_TIMEZONE = timezone(timedelta(hours=7))

#: tool ที่ดึงข้อมูลสดจากระบบจอง (ไม่ใช่คลังความรู้)
_LIVE_DATA_TOOLS = frozenset(
    {
        "search_available_rooms",
        "quote_price",
        "get_payment_info",
        "list_restaurants",
        "get_my_bookings",
        "get_booking_payment",
    }
)


def _clean_snippet(text: str, limit: int = 200) -> str:
    """ตัดบรรทัดบริบทออกแล้วย่อให้สั้นพอสำหรับแสดงเป็นแหล่งอ้างอิง"""
    lines = text.splitlines()
    if lines and lines[0].startswith(_CONTEXT_HEADER_PREFIXES) and lines[0].endswith("]"):
        lines = lines[1:]
    cleaned = " ".join(" ".join(lines).split())
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


def _today_in_thailand() -> str:
    return datetime.now(_THAI_TIMEZONE).strftime("%Y-%m-%d")


#: ข้อความที่ส่งกลับไปกระทุ้งเมื่อโมเดลปฏิเสธทั้งที่ยังไม่ได้ลองหาข้อมูลเลย
#:
#: บางรอบโมเดลตอบประโยคปฏิเสธทันทีโดยไม่เรียก tool สักตัว ซึ่งเป็นไปไม่ได้ในทางตรรกะ
#: เพราะมันยังไม่ได้ดูข้อมูลอะไรเลยจึงยังไม่มีสิทธิ์สรุปว่าไม่มีข้อมูล
#: เจอกับคำถามสั้น ๆ อย่าง "เรามา 3 วัน นอนเตียงเดี่ยว ราคาเท่าไหร่" ประมาณ 1 ใน 4 รอบ
#: การเขียนกติกาห้ามปฏิเสธไว้ใน system prompt ช่วยได้ไม่หมด เลยดักด้วยโค้ดอีกชั้น
#: กระทุ้งแค่ครั้งเดียวต่อหนึ่งคำถาม ถ้ายังยืนยันปฏิเสธก็ถือว่าปฏิเสธจริง
_NO_TOOL_REFUSAL_NUDGE = (
    "คุณยังไม่ได้เรียก tool ใด ๆ เลยในรอบนี้ จึงยังไม่มีข้อมูลพอที่จะสรุปว่าไม่มีข้อมูล "
    "ให้เรียก tool ที่เกี่ยวข้องกับคำถามก่อน แล้วค่อยตอบใหม่ "
    "ถ้าคำถามไม่ได้ระบุวันที่ ให้ใช้ช่วงวันนี้ถึงพรุ่งนี้"
)


def _is_refusal(answer_text: str, refusal: str) -> bool:
    """คำตอบนี้คือการปฏิเสธล้วน ๆ หรือเปล่า

    เดิมเช็คแค่ ``refusal in answer_text`` ซึ่งพอคำตอบสั้นและมาจากคลังความรู้อย่างเดียว
    ก็ใช้ได้ แต่พอบอทดึงข้อมูลสดมาตอบ คำตอบยาวขึ้นและเป็นหลายส่วน โมเดลจึงเริ่ม
    ตอบคำถามได้ครบแล้ว **ต่อท้าย** ด้วยประโยคปฏิเสธ (จับได้จากการทดสอบจริง
    ด้วยคำถาม "ห้องพักมีกี่แบบ ต่างกันยังไง") ผลคือ answered พลิกเป็น false
    ทั้งที่ตอบได้ แล้ว frontend จะขึ้นปุ่มโทรหารีสอร์ททับคำตอบที่ใช้งานได้จริง

    เกณฑ์ที่ถูกต้องคือ "เหลืออะไรอยู่บ้างไหมหลังตัดประโยคปฏิเสธออก"
    ถ้าไม่เหลือแปลว่าปฏิเสธจริง ถ้ายังเหลือแปลว่ามันตอบได้ แค่แถมประโยคปฏิเสธมาด้วย
    ซึ่งกรณีหลังเราปล่อยข้อความไว้ตามเดิม เพราะบางทีมันหมายถึง "ส่วนที่เหลือไม่มีข้อมูล"
    ซึ่งเป็นข้อมูลที่ลูกค้าควรได้รู้
    """
    if refusal not in answer_text:
        return False
    remainder = re.sub(r"[\W_]+", "", answer_text.replace(refusal, " "))
    return not remainder


class RAGPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
        booking_client: BookingAPIClient | None = None,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.llm_client = llm_client or get_llm_client()
        self.booking_client = booking_client or get_booking_api_client()

    async def answer(
        self,
        session_id: str,
        message: str,
        language: str = "th",
        resort_id: str | None = None,
        customer_token: str | None = None,
    ) -> ChatResponse:
        language = (
            language if language in settings.supported_languages else settings.default_language
        )
        refusal = no_context_answer(language)

        executor = ToolExecutor(
            booking_client=self.booking_client,
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            language=language,
            customer_token=customer_token,
        )

        messages: list[dict] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    language=language, refusal=refusal, today=_today_in_thailand()
                ),
            },
            {"role": "user", "content": message},
        ]

        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        answer_text = ""
        nudged = False

        for round_index in range(settings.agent_max_tool_rounds + 1):
            # รอบสุดท้ายไม่ส่ง tools ไปด้วย เพื่อบังคับให้โมเดลสรุปเป็นข้อความ
            is_final_round = round_index == settings.agent_max_tool_rounds
            result = await self.llm_client.generate(
                messages, tools=None if is_final_round else TOOL_SCHEMAS
            )

            if result.usage:
                for key in totals:
                    totals[key] += result.usage.get(key, 0)

            if not result.tool_calls:
                answer_text = result.text.strip()
                if (
                    not executor.called_tools
                    and not nudged
                    and not is_final_round
                    and _is_refusal(answer_text, refusal)
                ):
                    nudged = True
                    messages.append({"role": "assistant", "content": answer_text})
                    messages.append({"role": "user", "content": _NO_TOOL_REFUSAL_NUDGE})
                    continue
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": result.text or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in result.tool_calls
                    ],
                }
            )
            for call in result.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": await executor.run(call.name, call.arguments),
                    }
                )

        # โมเดลตอบข้อความว่างเปล่าได้ในบางกรณี ซึ่งแย่กว่าการปฏิเสธตรง ๆ
        if not answer_text:
            answer_text = refusal

        answered = not _is_refusal(answer_text, refusal)
        return self._build_response(
            session_id=session_id,
            answer_text=answer_text,
            answered=answered,
            executor=executor,
            totals=totals,
        )

    def _build_response(
        self,
        session_id: str,
        answer_text: str,
        answered: bool,
        executor: ToolExecutor,
        totals: dict[str, int],
    ) -> ChatResponse:
        # retrieval_score สะท้อนเฉพาะคุณภาพการค้นคลังความรู้ คำถามที่ตอบจากระบบจอง
        # ล้วน ๆ จึงได้ 0.0 ตามจริง ไม่ใช่เพราะค้นไม่เจอ
        best_distance = min(
            (hit["distance"] for hit in executor.knowledge_hits if hit["distance"] is not None),
            default=None,
        )
        retrieval_score = (
            max(0.0, min(1.0, 1.0 - best_distance)) if best_distance is not None else 0.0
        )

        # ข้อมูลจากระบบจองเป็นข้อมูลจริง ณ เวลานั้น ไม่ใช่การเดาจากความใกล้เคียงของเวกเตอร์
        # คำตอบที่อิงข้อมูลนั้นจึงไม่ควรถูกลดความมั่นใจลงตามคะแนน similarity
        used_live_data = any(name in _LIVE_DATA_TOOLS for name in executor.successful_tools)
        if not answered:
            confidence = 0.0
        elif used_live_data:
            confidence = 1.0
        else:
            confidence = retrieval_score

        # ตัด doc_id ซ้ำออก เพราะโมเดลค้นคลังความรู้ได้หลายรอบในหนึ่งคำถาม
        seen: set[str] = set()
        sources: list[Source] = []
        if answered:
            for hit in executor.knowledge_hits:
                if hit["doc_id"] in seen:
                    continue
                seen.add(hit["doc_id"])
                sources.append(
                    Source(doc_id=hit["doc_id"], snippet=_clean_snippet(hit["text"]))
                )

        return ChatResponse(
            session_id=session_id,
            answer=answer_text,
            sources=sources,
            answered=answered,
            confidence=round(confidence, 2),
            retrieval_score=round(retrieval_score, 4),
            suggested_action=SuggestedAction(type="none", url=""),
            tools_used=list(executor.called_tools),
            token_usage=TokenUsage(**totals) if totals["total_tokens"] else None,
        )


def get_rag_pipeline() -> RAGPipeline:
    """Factory function สำหรับใช้กับ FastAPI dependency injection"""
    return RAGPipeline()
