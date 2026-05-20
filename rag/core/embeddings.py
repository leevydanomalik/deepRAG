"""OpenAI embeddings (text-embedding-3-small, 1536-dim)."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from rag.core.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        api_key=s.openai_api_key,
    )


def embed_text(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)
