"""Centralized settings loaded from .env via pydantic-settings."""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM (OpenAI-compatible: DeepSeek, OpenAI, Z.ai, etc.)
    llm_api_key: SecretStr
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # Embeddings
    embedding_provider: str = "local"           # local | openai
    embedding_model_local: str = "BAAI/bge-small-en-v1.5"   # 384-dim
    embedding_model_openai: str = "text-embedding-3-small"  # 1536-dim
    embedding_dim: int = 384                    # set to 1536 if using openai

    # OpenAI key — only required when embedding_provider=openai
    openai_api_key: SecretStr | None = None

    # Postgres
    pg_dsn: str

    # Graph store (sqlite-graph extension)
    graph_db_path: str = "data/graph.db"
    graph_extension_path: str = "vendor/libgraph.dylib"

    # Loop RAG
    loop_max_iterations: int = 4
    loop_convergence_threshold: float = 0.15

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
