"""
เก็บประวัติบทสนทนาต่อ session ให้บอทตอบต่อเนื่องจากข้อความก่อนหน้าได้

**ทำไมเป็น SQLite ไม่ใช่ dict ใน memory** — แอปรันบน VM เครื่องเดียว uvicorn process เดียว
dict ก็ใช้ได้ แต่ทุกครั้งที่ ``docker compose up --build`` ประวัติจะหายหมด
ส่วน SQLite มากับ Python อยู่แล้วไม่ต้องลงอะไรเพิ่ม และ ``./data`` ถูก mount
ออกนอก container ไว้แล้ว (ดู docker-compose.yml) ไฟล์จึงอยู่รอดข้าม rebuild

**เก็บเฉพาะคำถามลูกค้ากับคำตอบสุดท้ายของบอท** ไม่เก็บผลของ tool และข้อความกระทุ้ง
- ผลของ tool (chunk จากคลังความรู้, JSON ห้องว่าง) คือก้อนที่กิน token มากที่สุด
  และประวัติถูกส่งซ้ำทุกรอบของลูป tool calling ถ้าเก็บไว้ค่า token จะบานเร็วมาก
- ข้อมูลห้องว่างกับราคาเปลี่ยนทุกครั้งที่มีคนจอง ถ้าเก็บผล tool ไว้ โมเดลอาจหยิบ
  ของเก่ามาตอบแทนที่จะเรียก tool ใหม่ การเหลือแค่ข้อความทำให้มันต้องไปดึงข้อมูลสดเสมอ

ทุก method เปิด connection ใหม่ทุกครั้ง เพราะ sqlite3 ห้ามใช้ connection ข้าม thread
และงานแต่ละครั้งเล็กมาก (ไม่ถึงมิลลิวินาที) จึงเรียกตรง ๆ ใน async ได้โดยไม่ต้องย้ายไป thread
"""

import sqlite3
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from app.config import settings

_SECONDS_PER_DAY = 24 * 60 * 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages (session_id, id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created ON chat_messages (created_at);
"""


class ChatHistoryStore:
    def __init__(
        self,
        path: str | Path | None = None,
        max_messages: int | None = None,
        ttl_days: int | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.path = Path(path or settings.chat_history_path)
        self.max_messages = max_messages or settings.chat_history_max_messages
        self.ttl_seconds = (ttl_days or settings.chat_history_ttl_days) * _SECONDS_PER_DAY
        #: รับ clock จากข้างนอกได้ เพื่อให้ test จำลองเวลาผ่านไป 7 วันได้โดยไม่ต้องรอจริง
        self.clock = clock

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _cutoff(self) -> float:
        return self.clock() - self.ttl_seconds

    def load(self, session_id: str) -> list[dict[str, str]]:
        """คืนประวัติล่าสุดไม่เกิน max_messages ข้อความ เรียงจากเก่าไปใหม่

        กรองข้อความที่หมดอายุออกตรงนี้ด้วย ไม่ได้พึ่งแค่การลบใน append
        เพราะ session ที่เงียบไปนานจะไม่มีใคร append ให้มันถูกลบ
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM chat_messages "
                "WHERE session_id = ? AND created_at >= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, self._cutoff(), self.max_messages),
            ).fetchall()

        history = [{"role": role, "content": content} for role, content in reversed(rows)]
        # ประวัติต้องเริ่มด้วยฝั่งลูกค้าเสมอ Gemini ไม่ยอมรับบทสนทนาที่ขึ้นต้นด้วยฝั่งโมเดล
        # เกิดได้ถ้าตั้ง max_messages เป็นเลขคี่ ซึ่งจะตัดกลางคู่ถาม-ตอบพอดี
        while history and history[0]["role"] != "user":
            history.pop(0)
        return history

    def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """บันทึกคู่ถาม-ตอบหนึ่งคู่ แล้วลบข้อความที่เก่าเกิน ttl ของทุก session ทิ้ง"""
        now = self.clock()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                [
                    (session_id, "user", user_message, now),
                    (session_id, "assistant", assistant_message, now),
                ],
            )
            conn.execute("DELETE FROM chat_messages WHERE created_at < ?", (self._cutoff(),))


@lru_cache
def get_chat_history_store() -> ChatHistoryStore:
    return ChatHistoryStore()
