"""FastAPI application entry point for the chatbot proxy."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.models import router as models_router
from app.core.config import get_settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("=== STARTUP SETTINGS ===")
    logger.info(f"base_url = {repr(settings.base_url)}")
    logger.info(f"api_key  = {repr(settings.api_key[:8] + '...' if settings.api_key else None)}")
    logger.info(f"retrieval_url = {repr(settings.retrieval_url)}")
    logger.info("========================")
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Chatbot Proxy API",
        description="OpenAI-compatible proxy with retrieval-augmented context injection",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(models_router)

    @app.get("/health")
    async def health_check():
        s = get_settings()
        return {"status": "healthy", "base_url": s.base_url, "has_api_key": bool(s.api_key)}

    return app


app = create_app()
