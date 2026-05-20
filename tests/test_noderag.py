"""End-to-end NodeRAG LangGraph test with mocked stores + LLM."""
from unittest.mock import MagicMock, patch

import networkx as nx

from rag.noderag.graph import build_graph


def _env(monkeypatch):
    for k in ("LLM_API_KEY", "PG_DSN"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_noderag_end_to_end(monkeypatch):
    _env(monkeypatch)

    fake_nr = MagicMock()
    fake_nr.vector_search.return_value = [
        {"id": "ent:abc", "node_type": "entity", "content": "LangGraph", "metadata": {}, "chunk_ids": ["c1"], "score": 0.9},
        {"id": "unit:xyz", "node_type": "semantic_unit", "content": "LangGraph supports cycles", "metadata": {}, "chunk_ids": ["c1"], "score": 0.85},
    ]
    fake_nr.fetch_by_ids.return_value = [
        {"id": "ent:abc", "node_type": "entity", "content": "LangGraph", "metadata": {}, "chunk_ids": ["c1"]},
        {"id": "unit:xyz", "node_type": "semantic_unit", "content": "LangGraph supports cycles", "metadata": {}, "chunk_ids": ["c1"]},
    ]

    fake_pg = MagicMock()
    fake_pg.fetch_by_ids.return_value = [
        {"id": "c1", "source": "a.md", "chunk_index": 0, "text": "LangGraph supports cycles for PDCA.", "metadata": {}},
    ]

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="LangGraph's cyclic feature enables PDCA loops.")

    fake_graph = nx.Graph()
    fake_graph.add_node("ent:abc"); fake_graph.add_node("unit:xyz")
    fake_graph.add_edge("ent:abc", "unit:xyz", weight=0.7)

    with patch("rag.noderag.graph.NodeRAGStore", return_value=fake_nr), \
         patch("rag.noderag.graph.PgStore", return_value=fake_pg), \
         patch("rag.noderag.graph.get_chat_model", return_value=fake_llm), \
         patch("rag.noderag.graph.embed_text", return_value=[0.0] * 384), \
         patch("rag.noderag.graph._build_ppr_graph", return_value=fake_graph):
        app = build_graph()
        out = app.invoke({"question": "What is LangGraph's cyclic feature?"})

    assert "PDCA" in out["answer"]
    assert len(out["seed_nodes"]) == 2
    assert len(out["ranked_nodes"]) >= 1
    nodes_visited = [h["node"] for h in out["history"]]
    assert nodes_visited == ["embed_query", "vector_seed", "ppr_propagate", "fetch_chunks", "generate"]
