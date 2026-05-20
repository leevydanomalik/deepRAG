from unittest.mock import patch

from rag.core.embeddings import get_embeddings


@patch("rag.core.embeddings.OpenAIEmbeddings")
def test_embeddings_configured(mock_emb, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")

    from rag.core.config import get_settings
    get_settings.cache_clear()

    get_embeddings()
    kwargs = mock_emb.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["api_key"].get_secret_value() == "sk-real"
