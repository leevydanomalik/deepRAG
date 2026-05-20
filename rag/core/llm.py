"""DeepSeek chat client. DeepSeek is OpenAI-compatible, so we use ChatOpenAI
with a custom base_url."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag.core.config import get_settings


def get_chat_model(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.deepseek_model,
        base_url=s.deepseek_base_url,
        api_key=s.deepseek_api_key,
        temperature=temperature,
        **kwargs,
    )
