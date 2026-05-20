"""OpenAI-compatible chat client (works with DeepSeek, OpenAI, Z.ai, …)."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag.core.config import get_settings


def get_chat_model(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        temperature=temperature,
        **kwargs,
    )
