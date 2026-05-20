from unittest.mock import MagicMock, patch

from rag.naive.graph import build_graph


def _env(monkeypatch):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "PG_DSN"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_naive_graph_invokes_retrieve_then_generate(monkeypatch):
    _env(monkeypatch)

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = [
        {"id": "c1", "source": "a.md", "chunk_index": 0, "text": "alpha", "metadata": {}, "score": 0.9},
        {"id": "c2", "source": "a.md", "chunk_index": 1, "text": "beta",  "metadata": {}, "score": 0.8},
    ]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="answer ok")

    with patch("rag.naive.graph.PgStore", return_value=fake_store), \
         patch("rag.naive.graph.embed_text", return_value=[0.0] * 384), \
         patch("rag.naive.graph.get_chat_model", return_value=fake_llm):
        app = build_graph()
        out = app.invoke({"question": "what is alpha?"})

    assert out["answer"] == "answer ok"
    assert len(out["retrieved"]) == 2
    assert out["history"][0]["node"] == "retrieve"
