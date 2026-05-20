from unittest.mock import MagicMock, patch

from rag.agentic.graph import build_graph


def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_agentic_graph_compiles(monkeypatch):
    _env(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.bind_tools.return_value = fake_llm
    fake_llm.invoke.return_value = MagicMock(content="done", tool_calls=[])
    with patch("rag.agentic.graph.get_chat_model", return_value=fake_llm):
        app = build_graph()
    assert app is not None
