"""
ประกอบ flow ของ RAG ทั้งหมดเข้าด้วยกัน:
รับคำถาม -> embed -> ค้นหา context จาก vector store -> ประกอบ prompt -> เรียก LLM -> คืนคำตอบ
"""

from app.core.prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT_TEMPLATE
from app.schemas.chat import ChatResponse, Source, SuggestedAction
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_client import LLMClient, get_llm_client
from app.services.vector_store import VectorStore, get_vector_store


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
        self, session_id: str, resort_id: str, message: str, language: str = "th"
    ) -> ChatResponse:
        # 1. แปลงคำถามเป็นเวกเตอร์
        query_embedding = self.embedding_service.embed_query(message)

        # 2. ค้นหา top-k context ที่ใกล้เคียงที่สุด
        hits = self.vector_store.query(query_embedding, resort_id=resort_id)

        if not hits:
            return ChatResponse(
                session_id=session_id,
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                confidence=0.0,
                suggested_action=SuggestedAction(type="none", url=""),
            )

        # 3. ประกอบ prompt จาก system prompt + context + คำถาม
        context_text = "\n\n".join(f"- {hit['text']}" for hit in hits)
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            resort_id=resort_id,
            language=language,
            context=context_text,
            question=message,
        )

        # 4. เรียก LLM เพื่อสร้างคำตอบ
        answer_text = await self.llm_client.generate(prompt)

        # 5. คำนวณ confidence คร่าว ๆ จากระยะห่างของผลลัพธ์อันดับ 1 (cosine distance ยิ่งน้อยยิ่งใกล้)
        best_distance = hits[0]["distance"]
        confidence = max(0.0, min(1.0, 1.0 - best_distance)) if best_distance is not None else 0.5

        sources = [Source(doc_id=hit["doc_id"], snippet=hit["text"][:200]) for hit in hits]

        return ChatResponse(
            session_id=session_id,
            answer=answer_text.strip(),
            sources=sources,
            confidence=round(confidence, 2),
            suggested_action=SuggestedAction(type="none", url=""),
        )


def get_rag_pipeline() -> RAGPipeline:
    """Factory function สำหรับใช้กับ FastAPI dependency injection"""
    return RAGPipeline()
