import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://pgdocs:pgdocs@localhost:5432/pgdocs")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # 384 dimensions
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "1") == "1"
    rerank_model: str = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    default_version: str = os.getenv("PG_DOCS_VERSION", "16")


settings = Settings()