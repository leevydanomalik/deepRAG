"""Test the build_noderag pipeline with mocked stores + LLM extractions."""
import json
from unittest.mock import MagicMock, patch


def _env(monkeypatch):
    for k in ("LLM_API_KEY", "PG_DSN"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_run_build_noderag_skip_communities(monkeypatch):
    _env(monkeypatch)
    from scripts.build_noderag import run_build_noderag

    fake_pg = MagicMock()
    fake_pg.conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ("c1", "src.md", 0, "LangGraph supports cycles. Authored by Harrison Chase."),
        ("c2", "src.md", 1, "LangChain Inc. maintains LangGraph and LangSmith."),
    ]

    # GraphStore.conn.execute(...).fetchall() returns entities or relations
    fake_gs = MagicMock()
    fake_gs.conn.execute.return_value.fetchall.side_effect = [
        # entities
        [
            ("langgraph", "LangGraph", json.dumps({"type": "Product", "description": "DAG framework", "source_chunks": ["c1"]})),
            ("harrison chase", "Harrison Chase", json.dumps({"type": "Person", "description": "creator", "source_chunks": ["c1"]})),
        ],
        # relations
        [
            ("langgraph", "harrison chase", "CREATED_BY", json.dumps({"description": "", "source_chunks": ["c1"]})),
        ],
    ]

    fake_nr = MagicMock()

    fake_llm = MagicMock()
    # First call extracts semantic_units for chunk c1, second extracts attributes for c1,
    # then same for c2.
    fake_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"semantic_units": [
            {"claim": "LangGraph supports cycles", "entities": ["LangGraph"]},
        ]})),
        MagicMock(content=json.dumps({"attributes": [
            {"entity": "LangGraph", "name": "creator", "value": "Harrison Chase"},
        ]})),
        MagicMock(content=json.dumps({"semantic_units": [
            {"claim": "LangChain Inc. maintains LangGraph", "entities": ["LangChain Inc."]},
        ]})),
        MagicMock(content=json.dumps({"attributes": []})),
    ]

    with patch("scripts.build_noderag.PgStore", return_value=fake_pg), \
         patch("scripts.build_noderag.GraphStore", return_value=fake_gs), \
         patch("scripts.build_noderag.NodeRAGStore", return_value=fake_nr), \
         patch("scripts.build_noderag.get_chat_model", return_value=fake_llm), \
         patch("scripts.build_noderag.embed_batch", side_effect=lambda xs: [[0.0] * 384 for _ in xs]):
        out = run_build_noderag(limit=None, skip_communities=True)

    assert out["entity"] == 2
    assert out["relationship"] == 1
    assert out["semantic_unit"] == 2
    assert out["attribute"] == 1
    assert out["high_level_element"] == 0
    assert out["high_level_overview"] == 0
    fake_nr.init_schema.assert_called_once()
    # 4 base types × at least one upsert call each
    assert fake_nr.upsert_node.call_count == 6
