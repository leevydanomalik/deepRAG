import json
from unittest.mock import MagicMock, patch

from rag.graph_rag.graph import build_graph


def _env(monkeypatch):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "PG_DSN"):
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
    fake_gs = MagicMock()
    fake_gs.expand_subgraph.return_value = (
        [{"name": "langgraph", "type": "Product", "description": "DAG framework", "source_chunks": ["c1"]}],
        [{"src": "langgraph", "dst": "langchain", "type": "BUILT_BY", "description": "", "source_chunks": ["c1"]}],
    )
    fake_pg = MagicMock()
    fake_pg.fetch_by_ids.return_value = [
        {"id": "c1", "source": "a.md", "chunk_index": 0, "text": "LangGraph...", "metadata": {}},
    ]

    with patch("rag.graph_rag.graph.get_chat_model", return_value=fake_llm), \
         patch("rag.graph_rag.graph.GraphStore", return_value=fake_gs), \
         patch("rag.graph_rag.graph.PgStore", return_value=fake_pg):
        app = build_graph()
        out = app.invoke({"question": "What is LangGraph?"})

    assert out["answer"] == "final answer"
    assert "LangGraph" in out["entities"]
    assert len(out["subgraph_nodes"]) == 1
