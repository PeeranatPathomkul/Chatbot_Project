"""
ประกอบ flow ของ RAG ทั้งหมดเข้าด้วยกัน:
รับคำถาม -> embed -> ค้นหา context จาก vector store -> ประกอบ prompt -> เรียก LLM -> คืนคำตอบ
"""

from app.config import settings
from app.core.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, no_context_answer
from app.schemas.chat import ChatResponse, Source, SuggestedAction
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStore, get_vector_store

#: ขึ้นต้นของบรรทัดบริบทที่ indexing-pipeline เติมไว้หัว chunk
#: (ดู chunking._build_context_header ในโปรเจกต์ indexing-pipeline)
#: บรรทัดนี้มีไว้ช่วย embedding ไม่ใช่ให้คนอ่าน จึงต้องตัดออกก่อนส่งกลับเป็น snippet
_CONTEXT_HEADER_PREFIXES = ("[หมวด:", "[Category:")


def _clean_snippet(text: str, limit: int = 200) -> str:
    """ตัดบรรทัดบริบทออกแล้วย่อให้สั้นพอสำหรับแสดงเป็นแหล่งอ้างอิง"""
    lines = text.splitlines()
    if lines and lines[0].startswith(_CONTEXT_HEADER_PREFIXES) and lines[0].endswith("]"):
        lines = lines[1:]
    cleaned = " ".join(" ".join(lines).split())
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")


class RAGPipeline:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.llm_client = llm_client or get_llm_client()

    async def answer(
        self,
        session_id: str,
        message: str,
        language: str = "th",
        resort_id: str | None = None,
    ) -> ChatResponse:
        # 1. แปลงคำถามเป็นเวกเตอร์ (embed_query เติม prefix "query: " ให้เอง)
        query_embedding = self.embedding_service.embed_query(message)

        # 2. ค้นหา top-k context ที่ใกล้เคียงที่สุด กรองตามภาษาที่ลูกค้าถาม
        language = (
            language if language in settings.supported_languages else settings.default_language
        )
        refusal = no_context_answer(language)
        hits = self.vector_store.query(query_embedding, language=language)

        if not hits:
            return ChatResponse(
                session_id=session_id,
                answer=refusal,
                sources=[],
                answered=False,
                confidence=0.0,
                retrieval_score=0.0,
                suggested_action=SuggestedAction(type="none", url=""),
            )

        # 3. ประกอบ prompt — กติกาไปไว้ใน system ส่วนข้อมูลอ้างอิงกับคำถามไว้ใน user
        #    การแยกแบบนี้เป็นสิ่งที่ทำให้โมเดลยอมปฏิเสธเมื่อไม่มีข้อมูล
        #    (ถ้ายัดรวมเป็นก้อนเดียวมันจะแต่งคำตอบขึ้นมา ดู app/core/prompts.py)
        context_text = "\n\n".join(f"- {hit['text']}" for hit in hits)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context_text, question=message)

        # 4. เรียก LLM เพื่อสร้างคำตอบ
        answer_text = (
            await self.llm_client.generate(
                user_prompt,
                system=SYSTEM_PROMPT.format(language=language, refusal=refusal),
            )
        ).strip()

        # 5. ประเมินว่าตอบได้จริงหรือปฏิเสธ
        #
        #    เดิมใช้ confidence = 1 - distance (ก็คือ cosine similarity ตรง ๆ)
        #    ซึ่งวัดจริงแล้วใช้ไม่ได้: คำถามที่คลังความรู้ "ไม่มีคำตอบ" ได้คะแนนเฉลี่ย
        #    0.835 ส่วนคำถามที่ตอบได้ 0.845 ต่างกันแค่ 0.011 (ดู scripts/evaluate.py
        #    ในโปรเจกต์ indexing-pipeline) การส่ง confidence 0.84 กลับไปพร้อมคำตอบ
        #    "ไม่มีข้อมูลเรื่องนี้ในระบบ" จึงขัดแย้งกันเองและทำให้ frontend ตัดสินใจผิด
        #
        #    ตัวชี้วัดที่เชื่อถือได้กว่าคือ "โมเดลยอมตอบหรือปฏิเสธ" เพราะมันสะท้อน
        #    ผลลัพธ์จริงหลังจากอ่าน context แล้ว
        best_distance = hits[0]["distance"]
        retrieval_score = (
            max(0.0, min(1.0, 1.0 - best_distance)) if best_distance is not None else 0.0
        )
        answered = refusal not in answer_text

        sources = (
            [
                Source(doc_id=hit["doc_id"], snippet=_clean_snippet(hit["text"]))
                for hit in hits
            ]
            if answered
            else []
        )

        return ChatResponse(
            session_id=session_id,
            answer=answer_text,
            sources=sources,
            answered=answered,
            confidence=round(retrieval_score, 2) if answered else 0.0,
            retrieval_score=round(retrieval_score, 4),
            suggested_action=SuggestedAction(type="none", url=""),
        )


def get_rag_pipeline() -> RAGPipeline:
    """Factory function สำหรับใช้กับ FastAPI dependency injection"""
    return RAGPipeline()
