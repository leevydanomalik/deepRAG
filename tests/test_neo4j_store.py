import os

import pytest

from rag.core.neo4j_store import Neo4jStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    if not os.environ.get("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD not set")
    s = Neo4jStore()
    s.init_schema()
    s.wipe()
    yield s
    s.close()


def test_merge_entity_idempotent(store):
    store.merge_entity(name="Acme", type_="Org", description="d1", chunk_ids=["c1"])
    store.merge_entity(name="Acme", type_="Org", description="d2", chunk_ids=["c2"])
    e = store.get_entity("Acme")
    assert e["type"] == "Org"
    assert "c1" in e["source_chunks"] and "c2" in e["source_chunks"]


def test_merge_relation(store):
    store.merge_entity("A", "Org", "", ["x"])
    store.merge_entity("B", "Person", "", ["x"])
    store.merge_relation("A", "B", "HAS_MEMBER", "A has member B", chunk_ids=["x"])
    rels = store.find_relations(["A"])
    assert any(r["type"] == "HAS_MEMBER" and r["dst"] == "b" for r in rels)


def test_expand_subgraph(store):
    store.merge_entity("A", "Org", "", [])
    store.merge_entity("B", "Person", "", [])
    store.merge_entity("C", "Place", "", [])
    store.merge_relation("A", "B", "EMPLOYS", "", chunk_ids=[])
    store.merge_relation("B", "C", "LIVES_IN", "", chunk_ids=[])
    nodes, edges = store.expand_subgraph(["A"], hops=2)
    names = {n["name"] for n in nodes}
    assert {"a", "b", "c"} <= names
