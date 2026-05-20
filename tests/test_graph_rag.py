import json
from unittest.mock import MagicMock, patch

from rag.graph_rag.graph import build_graph


def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_graph_rag_end_to_end(monkeypatch):
    _env(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"entities": ["LangGraph"]})),
        MagicMock(content="final answer"),
    ]
    fake_neo = MagicMock()
    fake_neo.expand_subgraph.return_value = (
        [{"name": "langgraph", "type": "Product", "description": "DAG framework", "source_chunks": ["c1"]}],
        [{"src": "langgraph", "dst": "langchain", "type": "BUILT_BY", "description": "", "source_chunks": ["c1"]}],
    )
    fake_pg = MagicMock()
    fake_pg.fetch_by_ids.return_value = [
        {"id": "c1", "source": "a.md", "chunk_index": 0, "text": "LangGraph...", "metadata": {}},
    ]

    with patch("rag.graph_rag.graph.get_chat_model", return_value=fake_llm), \
         patch("rag.graph_rag.graph.Neo4jStore", return_value=fake_neo), \
         patch("rag.graph_rag.graph.PgStore", return_value=fake_pg):
        app = build_graph()
        out = app.invoke({"question": "What is LangGraph?"})

    assert out["answer"] == "final answer"
    assert "LangGraph" in out["entities"]
    assert len(out["subgraph_nodes"]) == 1
