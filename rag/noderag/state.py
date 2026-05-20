"""State for NodeRAG: heterogeneous nodes + PPR-based retrieval."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class NodeRAGState(TypedDict, total=False):
    question: str
    q_embedding: list[float]            # embedded once, reused by seed step

    # shallow vector seeds
    seed_nodes: list[dict[str, Any]]    # top-K by cosine across all node types

    # deep PPR results
    ranked_nodes: list[dict[str, Any]]  # top-N after PPR
    ppr_scores: dict[str, float]        # node_id → score (for trace)

    # final chunk context
    retrieved: list[dict[str, Any]]

    answer: str
    history: Annotated[list[dict[str, Any]], add]
