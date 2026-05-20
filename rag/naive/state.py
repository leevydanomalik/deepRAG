"""State for Naive RAG: a single retrieve → generate pass."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class NaiveState(TypedDict, total=False):
    question: str
    retrieved: list[dict[str, Any]]
    answer: str
    history: Annotated[list[dict[str, Any]], add]
