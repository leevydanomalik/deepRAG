from rag.registry import REGISTRY, list_patterns


def test_registry_has_four_patterns():
    assert set(list_patterns()) == {"naive", "agentic", "graph", "loop"}
    for fn in REGISTRY.values():
        assert callable(fn)
