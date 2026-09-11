"""FastAPI application entrypoint"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, knowledge
from app.config import settings

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title=settings.app_name)

app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(knowledge.router, prefix=settings.api_prefix)

# หน้าเว็บทดสอบแชท เสิร์ฟจาก FastAPI เองเพื่อให้เป็น same-origin กับ API
# ถ้าเปิดไฟล์ HTML ตรง ๆ จาก file:// เบราว์เซอร์จะบล็อก fetch ด้วย CORS
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def chat_ui() -> FileResponse:
    """หน้าเว็บสำหรับทดลองคุยกับแชทบอท"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ping", include_in_schema=False)
async def ping() -> dict[str, str]:
    """เช็คว่าแอปยังตอบอยู่ (health check ของ chatbot อยู่ที่ /api/v1/chatbot/health)"""
    return {"message": f"{settings.app_name} is running"}
