"""
Endpoint สำหรับจัดการฐานความรู้ (knowledge base) แบบ manual ผ่าน API
สำหรับกรณีอยากเพิ่ม/แก้ไข/ลบความรู้ทีละรายการ โดยไม่ต้องรัน scripts/ingest.py ใหม่ทั้งหมด

หมายเหตุ: ในโครงนี้ implement เป็น skeleton เท่านั้น ยังไม่ได้ผูก auth/validation แบบเต็มรูปแบบ
"""

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class KnowledgeItem(BaseModel):
    resort_id: str
    category: str = ""
    question: str = ""
    answer: str
    source: str = ""


class KnowledgeCreateResponse(BaseModel):
    doc_id: str
    status: str = "created"


class KnowledgeDeleteResponse(BaseModel):
    doc_id: str
    status: str = "deleted"


@router.post("", response_model=KnowledgeCreateResponse)
async def create_knowledge(
    item: KnowledgeItem,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
) -> KnowledgeCreateResponse:
    """เพิ่มข้อมูลความรู้ใหม่ 1 รายการเข้า vector store"""
    doc_id = str(uuid.uuid4())
    text = f"{item.question}\n{item.answer}" if item.question else item.answer
    embedding = embedding_service.embed_documents([text])[0]
    vector_store.add_documents(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[
            {
                "resort_id": item.resort_id,
                "category": item.category,
                "source": item.source,
            }
        ],
    )
    return KnowledgeCreateResponse(doc_id=doc_id)


@router.put("/{doc_id}", response_model=KnowledgeCreateResponse)
async def update_knowledge(
    doc_id: str,
    item: KnowledgeItem,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
) -> KnowledgeCreateResponse:
    """แก้ไขข้อมูลความรู้ที่มีอยู่แล้ว (upsert ตาม doc_id เดิม)"""
    text = f"{item.question}\n{item.answer}" if item.question else item.answer
    embedding = embedding_service.embed_documents([text])[0]
    vector_store.add_documents(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[
            {
                "resort_id": item.resort_id,
                "category": item.category,
                "source": item.source,
            }
        ],
    )
    return KnowledgeCreateResponse(doc_id=doc_id, status="updated")


@router.delete("/{doc_id}", response_model=KnowledgeDeleteResponse)
async def delete_knowledge(
    doc_id: str,
    vector_store: VectorStore = Depends(get_vector_store),
) -> KnowledgeDeleteResponse:
    """ลบข้อมูลความรู้ตาม doc_id"""
    vector_store.delete(ids=[doc_id])
    return KnowledgeDeleteResponse(doc_id=doc_id)
