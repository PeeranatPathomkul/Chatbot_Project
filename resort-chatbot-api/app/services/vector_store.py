"""ห่อ ChromaDB (embedded mode, persist ลง local disk) สำหรับเก็บและค้นหาความรู้"""

from functools import lru_cache
from typing import Any

import chromadb

from app.config import settings


class VectorStore:
    def __init__(self, path: str | None = None, collection_name: str | None = None):
        self.client = chromadb.PersistentClient(path=path or settings.chroma_path)
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

    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        resort_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """ค้นหาเอกสารที่ใกล้เคียงที่สุดกับ query_embedding

        ถ้าระบุ resort_id จะกรองเฉพาะเอกสารของรีสอร์ตนั้น (metadata filter)
        """
        where = {"resort_id": resort_id} if resort_id else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k or settings.retrieval_top_k,
            where=where,
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
