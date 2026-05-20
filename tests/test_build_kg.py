import json
from unittest.mock import MagicMock, patch

from scripts.build_kg import extract_kg_from_chunk, run_build_kg


def test_extract_kg_from_chunk_parses_json():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "entities": [{"name": "LangGraph", "type": "Product", "description": "DAG framework"}],
        "relations": [{"src": "LangGraph", "dst": "LangChain", "type": "BUILT_BY", "description": ""}],
    }))
    out = extract_kg_from_chunk(fake_llm, "irrelevant text")
    assert out["entities"][0]["name"] == "LangGraph"
    assert out["relations"][0]["type"] == "BUILT_BY"


def test_extract_kg_handles_malformed_json():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="not json at all")
    out = extract_kg_from_chunk(fake_llm, "x")
    assert out == {"entities": [], "relations": []}


def test_run_build_kg_iterates_chunks(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()

    fake_pg = MagicMock()
    fake_pg.conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ("c1", "src.md", 0, "LangGraph builds DAGs.", {}),
    ]
    fake_neo = MagicMock()
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "entities": [{"name": "LangGraph", "type": "Product", "description": ""}],
        "relations": [],
    }))
    with patch("scripts.build_kg.PgStore", return_value=fake_pg), \
         patch("scripts.build_kg.Neo4jStore", return_value=fake_neo), \
         patch("scripts.build_kg.get_chat_model", return_value=fake_llm):
        e, r = run_build_kg(limit=1)
    assert e == 1 and r == 0
    fake_neo.init_schema.assert_called_once()
    fake_neo.merge_entity.assert_called()
