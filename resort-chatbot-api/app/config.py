"""
โหลดค่า config ทั้งหมดจาก environment variables (.env) ด้วย pydantic-settings
ทุกส่วนของแอปควร import ตัวแปร `settings` จากไฟล์นี้ ห้าม hardcode ค่า config ที่อื่น
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider ---
    llm_provider: str = "typhoon"  # "typhoon" หรือ "gemini"

    typhoon_api_key: str = ""
    typhoon_base_url: str = "https://api.opentyphoon.ai/v1"
    typhoon_model: str = "typhoon-v2.1-12b-instruct"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # --- Embedding ---
    embedding_model: str = "intfloat/multilingual-e5-base"

    # --- Vector store ---
    chroma_path: str = "./data/chroma"
    chroma_collection_name: str = "resort_knowledge"

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
