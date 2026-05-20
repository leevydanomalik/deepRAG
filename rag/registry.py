"""Pattern name → graph builder. Single entry point used by CLI/API/UI."""
from __future__ import annotations

from typing import Callable

from rag.agentic.graph import build_graph as build_agentic
from rag.graph_rag.graph import build_graph as build_graph_rag
from rag.loop.graph import build_graph as build_loop
from rag.naive.graph import build_graph as build_naive
from rag.noderag.graph import build_graph as build_noderag

REGISTRY: dict[str, Callable] = {
    "naive": build_naive,
    "agentic": build_agentic,
    "graph": build_graph_rag,
    "loop": build_loop,
    "noderag": build_noderag,
}


def list_patterns() -> list[str]:
    return list(REGISTRY.keys())


def run(pattern: str, question: str) -> dict:
    if pattern not in REGISTRY:
        raise ValueError(f"unknown pattern: {pattern!r} (have {list_patterns()})")
    app = REGISTRY[pattern]()
    initial = {"question": question, "iteration": 0} if pattern == "loop" else {"question": question}
    result = app.invoke(initial)
    return {
        "answer": result.get("answer", ""),
        "trace": result.get("history", []),
        "raw": {k: v for k, v in result.items() if k not in {"answer", "history"}},
    }
