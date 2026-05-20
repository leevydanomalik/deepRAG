"""LangGraph state for Loop RAG (PDCA closed-loop)."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class Intent(TypedDict, total=False):
    task_type: str
    entities: list[str]
    constraints: list[str]
    sub_goals: list[str]


class PromptConfig(TypedDict, total=False):
    template: str


class Evidence(TypedDict, total=False):
    id: str
    source: str
    chunk_index: int
    text: str
    score: float
    support: float


class Deviation(TypedDict, total=False):
    align: float
    faith: float
    constraint: float
    total: float
    dominant: str
    converged: bool


class LoopState(TypedDict, total=False):
    user_input: str
    question: str

    intent: Intent
    prompt_config: PromptConfig

    evidence: list[Evidence]
    answer: str

    deviation: Deviation

    iteration: int
    converged: bool

    history: Annotated[list[dict[str, Any]], add]
