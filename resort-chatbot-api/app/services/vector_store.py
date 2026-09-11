"""ห่อ ChromaDB (embedded mode, persist ลง local disk) สำหรับเก็บและค้นหาความรู้"""

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings


class VectorStore:
    def __init__(self, path: str | None = None, collection_name: str | None = None):
        # ปิด telemetry เพราะ chromadb 0.5.5 กับ posthog เวอร์ชันใหม่เข้ากันไม่ได้
        # แล้วพ่น "Failed to send telemetry event" รกล็อกทุกครั้งที่เรียกใช้งาน
        self.client = chromadb.PersistentClient(
            path=path or settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name or settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """เพิ่มเอกสาร (พร้อม embedding ที่คำนวณไว้แล้ว) ลง collection"""
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    @staticmethod
    def _build_where(language: str | None, category: str | None) -> dict[str, Any] | None:
        """ประกอบ metadata filter ของ Chroma

        Chroma รับเงื่อนไขเดียวเป็น dict ธรรมดา แต่หลายเงื่อนไขต้องห่อด้วย ``$and``
        """
        clauses = [
            {field: value}
            for field, value in (("language", language), ("category", category))
            if value
        ]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        language: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """ค้นหาเอกสารที่ใกล้เคียงที่สุดกับ query_embedding

        Args:
            query_embedding: เวกเตอร์ของคำถาม (ต้องมาจาก embed_query เท่านั้น)
            top_k: จำนวนผลลัพธ์
            language: กรองเฉพาะภาษา "th" / "en" — **ควรส่งเสมอ**
                เพราะ multilingual-e5 จับคู่ข้ามภาษาได้ ลูกค้าที่ถามไทย
                จึงอาจได้ chunk ภาษาอังกฤษกลับไปแล้ว LLM ตอบผิดภาษา
            category: กรองเฉพาะหมวด เช่น "FAQ" (ปกติไม่ต้องใส่)

        หมายเหตุ: เดิมฟังก์ชันนี้กรองด้วย ``resort_id`` ซึ่งเป็นการออกแบบสำหรับ
        multi-tenant แต่คลังความรู้ปัจจุบันเป็นรีสอร์ทเดียวและไม่มี field นั้นใน
        metadata เลย การกรองด้วย resort_id จึงคืนผลลัพธ์ว่างเสมอโดยไม่มี error
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k or settings.retrieval_top_k,
            where=self._build_where(language, category),
        )

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                {
                    "doc_id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return hits

    def delete(self, ids: list[str]) -> None:
        """ลบเอกสารออกจาก collection ตาม id"""
        self.collection.delete(ids=ids)


@lru_cache
def get_vector_store() -> VectorStore:
    """คืนค่า VectorStore แบบ cache ไว้ตัวเดียว"""
    return VectorStore()
