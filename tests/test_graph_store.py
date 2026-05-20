"""Real-extension tests for GraphStore — load libgraph.dylib and round-trip
entities/relations in a tempfile DB. Skips if the extension isn't built."""
from __future__ import annotations

import os

import pytest

from rag.core.graph_store import GraphStore


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


@pytest.fixture
def store(tmp_path, monkeypatch):
    ext = "vendor/libgraph.dylib"
    if not os.path.exists(ext):
        pytest.skip(f"sqlite-graph extension not built at {ext}")
    db = tmp_path / "graph.db"
    s = GraphStore(db_path=str(db), extension_path=ext)
    s.init_schema()
    yield s
    s.close()


def test_merge_entity_idempotent(store):
    store.merge_entity("Acme", "Org", "first description", ["c1"])
    store.merge_entity("Acme", "Org", "second note", ["c2"])
    e = store.get_entity("Acme")
    assert e["name"] == "acme"
    assert e["display_name"] == "Acme"
    assert "first description" in e["description"]
    assert "second note" in e["description"]
    assert set(e["source_chunks"]) >= {"c1", "c2"}


def test_merge_relation(store):
    store.merge_entity("A", "Org", "", ["x"])
    store.merge_entity("B", "Person", "", ["x"])
    store.merge_relation("A", "B", "HAS_MEMBER", "A has member B", ["x"])
    rels = store.find_relations(["A"])
    assert any(r["type"] == "HAS_MEMBER" and r["dst"] == "b" for r in rels)


def test_expand_subgraph_two_hops(store):
    store.merge_entity("A", "Org", "", [])
    store.merge_entity("B", "Person", "", [])
    store.merge_entity("C", "Place", "", [])
    store.merge_entity("D", "Item", "", [])
    store.merge_relation("A", "B", "EMPLOYS", "", [])
    store.merge_relation("B", "C", "LIVES_IN", "", [])
    store.merge_relation("C", "D", "CONTAINS", "", [])  # 3 hops away from A

    nodes, edges = store.expand_subgraph(["A"], hops=2)
    names = {n["name"] for n in nodes}
    assert {"a", "b", "c"} <= names
    assert "d" not in names  # 3 hops away, beyond hops=2


def test_relation_weight_increments(store):
    store.merge_relation("X", "Y", "REL", "", [])
    store.merge_relation("X", "Y", "REL", "", [])
    rels = store.find_relations(["X"])
    matching = [r for r in rels if r["type"] == "REL" and r["dst"] == "y"]
    assert len(matching) == 1  # not duplicated as a second edge


def test_find_entities_by_names(store):
    store.merge_entity("Alpha", "Concept", "first", [])
    store.merge_entity("Beta", "Concept", "second", [])
    out = store.find_entities_by_names(["Alpha", "Beta", "Gamma"])
    names = {e["name"] for e in out}
    assert names == {"alpha", "beta"}


def test_get_entity_missing(store):
    assert store.get_entity("NoSuch") is None


def test_wipe(store):
    store.merge_entity("A", "Org", "", [])
    store.merge_entity("B", "Org", "", [])
    store.merge_relation("A", "B", "R", "", [])
    store.wipe()
    assert store.get_entity("A") is None
    assert store.find_relations(["A"]) == []
