from unittest.mock import MagicMock, patch

from rag.core.llm import get_chat_model


@patch("rag.core.llm.ChatOpenAI")
def test_chat_model_configured_for_deepseek(mock_chat, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("PG_DSN", "x")

    from rag.core.config import get_settings
    get_settings.cache_clear()

    get_chat_model()

    mock_chat.assert_called_once()
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["base_url"] == "https://api.deepseek.com/v1"
    assert kwargs["api_key"].get_secret_value() == "sk-test"
    assert kwargs["temperature"] == 0.0
