"""Embeddings — supports a local sentence-transformers model OR OpenAI.

Switch via EMBEDDING_PROVIDER=local|openai. Local default is
BAAI/bge-small-en-v1.5 (384-dim), downloaded once on first use.
"""
from __future__ import annotations

from functools import lru_cache

from rag.core.config import get_settings


class _LocalEmbedder:
    """Minimal embed_query/embed_documents wrapper over sentence-transformers."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vecs]


@lru_cache
def get_embeddings():
    """Return an embedder with .embed_query(str) and .embed_documents(list[str])."""
    s = get_settings()
    if s.embedding_provider == "local":
        return _LocalEmbedder(s.embedding_model_local)
    # openai
    from langchain_openai import OpenAIEmbeddings

    if s.openai_api_key is None:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
        )
    return OpenAIEmbeddings(
        model=s.embedding_model_openai,
        api_key=s.openai_api_key,
    )


def embed_text(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)
