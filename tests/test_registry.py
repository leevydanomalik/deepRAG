from rag.registry import REGISTRY, list_patterns


def test_registry_has_five_patterns():
    assert set(list_patterns()) == {"naive", "agentic", "graph", "loop", "noderag"}
    for fn in REGISTRY.values():
        assert callable(fn)
