"""State for Graph RAG: entity extract → subgraph expand → chunk lookup → generate."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class GraphRAGState(TypedDict, total=False):
    question: str
    entities: list[str]
    subgraph_nodes: list[dict[str, Any]]
    subgraph_edges: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    answer: str
    history: Annotated[list[dict[str, Any]], add]
