"""
Centralized configuration using Pydantic BaseSettings.
Reads from environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── API Info ──
    APP_TITLE: str = "ThyraX CDSS API"
    APP_VERSION: str = "2.0.0"

    # ── LLM ──
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1

    # ── Database ──
    DATABASE_URL: str = "sqlite+aiosqlite:///./thyrax.db"

    # ── ChromaDB ──
    CHROMA_PERSIST_DIR: str = str(
        Path(__file__).resolve().parent.parent.parent / "data"
    )
    CHROMA_GUIDELINES_COLLECTION: str = "pdf_documents"
    CHROMA_SIMILAR_CASES_COLLECTION: str = "similar_cases"

    # ── Embeddings ──
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
