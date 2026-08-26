"""
โมดูลแปลงข้อความเป็นเวกเตอร์ด้วย ``intfloat/multilingual-e5-base`` (768 มิติ)

กฎเหล็กของโมเดลตระกูล e5
------------------------
โมเดล e5 ถูกเทรนแบบ asymmetric คือ "เอกสาร" กับ "คำถาม" ต้องมี prefix คนละอัน::

    เอกสารที่ index   ->  "passage: " + ข้อความ
    คำถามของผู้ใช้     ->  "query: "   + ข้อความ

ถ้าลืมใส่ prefix หรือใส่สลับกัน similarity จะเพี้ยนอย่างเห็นได้ชัด (ผลทดสอบใน
``tests/test_embedding.py`` ยืนยันว่า prefix ถูกต้องให้คะแนนสูงกว่าอย่างมีนัยสำคัญ)
โมดูลนี้จึงแยกเป็น 2 ฟังก์ชันชัดเจน และเป็นที่เดียวที่แตะ prefix เพื่อกันเรียกผิด

หมายเหตุเรื่อง normalize
------------------------
เวกเตอร์ถูก normalize ให้ยาว 1 เสมอ (``config.NORMALIZE_EMBEDDINGS``) ทำให้
dot product เท่ากับ cosine similarity พอดี และเข้าคู่กับ ChromaDB ที่ตั้ง
``hnsw:space = "cosine"`` ไว้ใน ``indexer.py``
"""

from __future__ import annotations

import time
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src import config
from src.logging_utils import format_duration, get_logger

logger = get_logger(__name__)

#: type alias ให้อ่านง่ายขึ้น
Vector = list[float]


class ThaiEmbedder:
    """ห่อ SentenceTransformer พร้อมจัดการ prefix ของ e5 และการทำ batch ให้อัตโนมัติ"""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE

        logger.info("กำลังโหลดโมเดล embedding: %s (ครั้งแรกอาจต้องดาวน์โหลดสักครู่)", self.model_name)
        started = time.perf_counter()
        self._model = SentenceTransformer(
            self.model_name,
            device=device or config.EMBEDDING_DEVICE,
        )
        logger.info(
            "โหลดโมเดลเสร็จใน %s | มิติเวกเตอร์ = %d | อุปกรณ์ = %s",
            format_duration(time.perf_counter() - started),
            self.dimension,
            self._model.device,
        )

    # ------------------------------------------------------------------
    # คุณสมบัติของโมเดล
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """จำนวนมิติของเวกเตอร์ที่โมเดลผลิต (ควรได้ 768 สำหรับ e5-base)"""
        return self._model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        """ความยาวสูงสุด (token) ที่โมเดลรับได้ ข้อความที่ยาวกว่านี้จะถูกตัดทิ้ง"""
        return self._model.max_seq_length

    def count_tokens(self, text: str) -> int:
        """นับจำนวน token ของข้อความตาม tokenizer ของโมเดล

        ใช้ตรวจว่า chunk_size ที่ตั้งไว้ (นับเป็นตัวอักษร) แปลงเป็น token แล้วยัง
        ไม่เกินเพดาน 512 token ของ e5-base
        """
        return len(self._model.tokenizer.encode(text, add_special_tokens=True))

    # ------------------------------------------------------------------
    # การ embed
    # ------------------------------------------------------------------

    def embed_documents(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[Vector]:
        """แปลง "เอกสาร/chunk" เป็นเวกเตอร์ — เติม prefix ``"passage: "`` ให้อัตโนมัติ

        ประมวลผลเป็น batch ตาม ``config.EMBED_BATCH_SIZE`` เพื่อความเร็วและกัน RAM บาน

        Args:
            texts: list ของข้อความดิบ (ห้ามใส่ prefix มาเอง เดี๋ยวจะซ้อนกัน)
            show_progress: แสดง progress bar ของ sentence-transformers

        Returns:
            list ของเวกเตอร์ ความยาวเท่ากับ ``texts`` และเรียงลำดับตรงกัน
        """
        if not texts:
            return []

        prefixed = [config.PASSAGE_PREFIX + text for text in texts]
        started = time.perf_counter()
        vectors = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        logger.debug(
            "embed เอกสาร %d ชิ้น ใช้เวลา %s",
            len(texts),
            format_duration(time.perf_counter() - started),
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> Vector:
        """แปลง "คำถามของผู้ใช้" เป็นเวกเตอร์ — เติม prefix ``"query: "`` ให้อัตโนมัติ"""
        vector = self._model.encode(
            config.QUERY_PREFIX + text,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def embed_queries(self, texts: list[str]) -> list[Vector]:
        """แปลงคำถามหลายข้อพร้อมกันเป็น batch (ใช้ในสคริปต์ทดสอบ)"""
        if not texts:
            return []
        prefixed = [config.QUERY_PREFIX + text for text in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        return vectors.tolist()


@lru_cache(maxsize=1)
def get_embedder() -> ThaiEmbedder:
    """คืน ThaiEmbedder ตัวเดียวที่ใช้ร่วมกันทั้งโปรเซส

    การโหลดโมเดลใช้เวลาหลายวินาทีและกินหน่วยความจำหลายร้อย MB จึงต้อง cache ไว้
    """
    return ThaiEmbedder()


def cosine_similarity(a: Vector, b: Vector) -> float:
    """คำนวณ cosine similarity ระหว่างสองเวกเตอร์ (ใช้ในเทสต์เพื่อไม่ต้องพึ่ง Chroma)"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
