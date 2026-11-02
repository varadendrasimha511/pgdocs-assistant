import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # reads .env when you run outside Docker


@dataclass(frozen=True)
class Settings:
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://pgdocs:pgdocs@127.0.0.1:5433/pgdocs"
    )

    # Embeddings and reranker (run locally, no API cost)
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384 dimensions
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "1") == "1"
    rerank_model: str = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

    # LLM: gemini | groq | openrouter | ollama | anthropic
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")  # optional override
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

    # Which PostgreSQL docs version to search by default
    default_version: str = os.getenv("PG_DOCS_VERSION", "16")


settings = Settings()