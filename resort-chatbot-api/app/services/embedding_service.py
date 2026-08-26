"""
ห่อ sentence-transformers ให้ใช้งานง่าย
ใช้โมเดล multilingual-e5-base ซึ่งรันโลคัลได้ฟรี ไม่ต้องเรียก API ภายนอก

หมายเหตุ: โมเดลตระกูล e5 ต้องการ prefix "query: " หรือ "passage: " นำหน้าข้อความ
ก่อนทำ embedding ตาม convention ของโมเดล (ดู model card บน HuggingFace)
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        """แปลงคำถามของผู้ใช้เป็นเวกเตอร์"""
        return self._model.encode(f"query: {text}", normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """แปลงเอกสาร/chunk เป็นเวกเตอร์ (ใช้ตอน ingest)"""
        prefixed = [f"passage: {t}" for t in texts]
        return self._model.encode(prefixed, normalize_embeddings=True).tolist()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """คืนค่า EmbeddingService แบบ cache ไว้ตัวเดียว เพราะการโหลดโมเดลใช้เวลาและหน่วยความจำ"""
    return EmbeddingService()
