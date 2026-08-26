"""FastAPI application entrypoint"""

from fastapi import FastAPI

from app.api.routes import chat, knowledge
from app.config import settings

app = FastAPI(title=settings.app_name)

app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(knowledge.router, prefix=settings.api_prefix)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}
