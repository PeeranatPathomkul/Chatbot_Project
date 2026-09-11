"""
โหลดค่า config ทั้งหมดจาก environment variables (.env) ด้วย pydantic-settings
ทุกส่วนของแอปควร import ตัวแปร `settings` จากไฟล์นี้ ห้าม hardcode ค่า config ที่อื่น
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: root ของ resort-chatbot-api (โฟลเดอร์ที่มี app/ และ .env)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: ที่เก็บ vector store — อยู่ในโปรเจกต์ indexing-pipeline ซึ่งเป็นคนสร้างมัน
#: API เป็นแค่ผู้อ่าน ไม่ได้เป็นคนเขียน collection นี้
DEFAULT_CHROMA_PATH = PROJECT_ROOT.parent / "indexing-pipeline" / "chroma_db"


class Settings(BaseSettings):
    # ระบุ path ของ .env แบบเต็ม ไม่ใช่ ".env" เฉย ๆ
    # เพราะ pydantic-settings มองหาไฟล์เทียบกับ current working directory
    # ถ้ารัน uvicorn จากโฟลเดอร์อื่น จะอ่าน .env ไม่เจอแล้วได้ api_key ว่าง
    # อาการคือ error "Illegal header value b'Bearer '" ซึ่งหาสาเหตุยากมาก
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM provider ---
    llm_provider: str = "typhoon"  # "typhoon" หรือ "gemini"

    typhoon_api_key: str = ""
    typhoon_base_url: str = "https://api.opentyphoon.ai/v1"
    typhoon_model: str = "typhoon-v2.5-30b-a3b-instruct"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # --- Embedding ---
    embedding_model: str = "intfloat/multilingual-e5-base"

    # --- Vector store ---
    chroma_path: str = str(DEFAULT_CHROMA_PATH)
    chroma_collection_name: str = "resort_knowledge"

    # --- ภาษาที่คลังความรู้รองรับ (ต้องตรงกับโฟลเดอร์ย่อยใน indexing-pipeline/data/) ---
    supported_languages: tuple[str, ...] = ("th", "en")
    default_language: str = "th"

    # --- RAG ---
    retrieval_top_k: int = 3
    llm_timeout_seconds: int = 30

    # --- App ---
    app_name: str = "Resort Chatbot API"
    api_prefix: str = "/api/v1"


@lru_cache
def get_settings() -> Settings:
    """คืนค่า Settings แบบ cache ไว้ครั้งเดียว (อ่าน .env ครั้งเดียวพอ)"""
    return Settings()


settings = get_settings()
