from unittest.mock import MagicMock, patch

from rag.core import embeddings as emb_mod
from rag.core.embeddings import get_embeddings


def _clear(monkeypatch):
    from rag.core.config import get_settings
    get_settings.cache_clear()
    get_embeddings.cache_clear()


def test_embeddings_local_uses_sentence_transformers(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    _clear(monkeypatch)

    with patch.object(emb_mod, "_LocalEmbedder") as mock_local:
        get_embeddings()
    mock_local.assert_called_once_with("BAAI/bge-small-en-v1.5")


def test_embeddings_openai_path(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    _clear(monkeypatch)

    fake_openai_cls = MagicMock()
    with patch.dict("sys.modules", {"langchain_openai": MagicMock(OpenAIEmbeddings=fake_openai_cls)}):
        get_embeddings()
    fake_openai_cls.assert_called_once()
    kwargs = fake_openai_cls.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["api_key"].get_secret_value() == "sk-real"


def test_embeddings_openai_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _clear(monkeypatch)

    import pytest
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_embeddings()
