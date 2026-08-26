"""
ตัวช่วยเรื่อง logging และการพิมพ์ภาษาไทยบน Windows console

เหตุผลที่ต้องมีไฟล์นี้: console ของ Windows ค่าเริ่มต้นเป็น cp874/cp1252 การ print
ข้อความไทยจะโยน UnicodeEncodeError ทันที จึงต้อง reconfigure stdout/stderr เป็น UTF-8
ก่อนพิมพ์อะไรก็ตาม ทุก entry point ของโปรเจกต์นี้เรียก ``setup_logging()`` เป็นอย่างแรก
"""

from __future__ import annotations

import logging
import sys

from src import config


def force_utf8_stdout() -> None:
    """บังคับ stdout/stderr เป็น UTF-8 เพื่อให้พิมพ์ภาษาไทยได้บนทุกแพลตฟอร์ม"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(level: str | None = None) -> None:
    """ตั้งค่า root logger ครั้งเดียวสำหรับทั้งสคริปต์"""
    force_utf8_stdout()
    logging.basicConfig(
        level=(level or config.LOG_LEVEL).upper(),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    # sentence-transformers / chromadb คุยเยอะเกินไปในระดับ INFO
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    # chromadb 0.5.5 เข้ากับ posthog เวอร์ชันใหม่ไม่ได้ แล้วพ่น ERROR
    # "Failed to send telemetry event ..." รกล็อกทุกครั้ง ทั้งที่ไม่กระทบการทำงาน
    # (ปิด telemetry ที่ต้นทางแล้วใน indexer.get_client ตรงนี้เป็นตาข่ายกันอีกชั้น)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


def get_logger(name: str) -> logging.Logger:
    """คืน logger ที่ตั้งชื่อสั้น ๆ ตามโมดูล"""
    return logging.getLogger(name)


def format_duration(seconds: float) -> str:
    """แปลงวินาทีเป็นข้อความอ่านง่าย เช่น '1.4 วินาที' หรือ '2 นาที 5.0 วินาที'"""
    if seconds < 60:
        return f"{seconds:.1f} วินาที"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} นาที {remainder:.1f} วินาที"
