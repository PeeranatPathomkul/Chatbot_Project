"""
ห่อ sentence-transformers ให้ใช้งานง่าย
ใช้โมเดล multilingual-e5-base ซึ่งรันโลคัลได้ฟรี ไม่ต้องเรียก API ภายนอก

หมายเหตุ: โมเดลตระกูล e5 ต้องการ prefix "query: " หรือ "passage: " นำหน้าข้อความ
ก่อนทำ embedding ตาม convention ของโมเดล (ดู model card บน HuggingFace)
"""

import threading

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


_load_lock = threading.Lock()
_instance: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """คืนค่า EmbeddingService ตัวเดียวทั้งแอป เพราะการโหลดโมเดลใช้เวลาและหน่วยความจำ

    **ต้องโหลดทีละ thread** — โมเดลถูกโหลดตอน request แรก เดิมใช้ lru_cache ซึ่งไม่กันการเรียกซ้อน
    ถ้ามีหลาย request เข้ามาพร้อมกันตอน container เพิ่งเปิด ทุก thread จะโหลดโมเดลพร้อมกัน
    แล้ว transformers พังด้วย "Cannot copy out of meta tensor" และพังค้างทุก request
    จนกว่าจะ restart (เจอจริงตอนยิงทดสอบ 4 session พร้อมกันหลัง rebuild: 500 ทั้ง 175 ข้อความ)
    เช็คซ้ำหลังได้ lock เพื่อให้ thread ที่รออยู่ได้ตัวที่โหลดเสร็จแล้ว ไม่โหลดซ้ำอีกรอบ
    """
    global _instance
    if _instance is None:
        with _load_lock:
            if _instance is None:
                _instance = EmbeddingService()
    return _instance
