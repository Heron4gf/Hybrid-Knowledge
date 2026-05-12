"""Configuration management for the chatbot proxy."""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    retrieval_url: str = "http://localhost:8001/v1"


def get_settings() -> Settings:
    """Get settings instance, always reading from environment."""
    return Settings()
