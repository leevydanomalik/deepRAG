"""State for Agentic RAG: agent + tools loop."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class AgenticState(TypedDict, total=False):
    question: str
    messages: Annotated[list[BaseMessage], add]
    answer: str
    history: Annotated[list[dict[str, Any]], add]
