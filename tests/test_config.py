import os
from rag.core.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("PG_DSN", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")

    s = Settings()
    assert s.deepseek_api_key.get_secret_value() == "sk-test"
    assert s.deepseek_model == "deepseek-chat"
    assert s.embedding_dim == 1536
    assert s.loop_max_iterations == 4
    assert s.loop_convergence_threshold == 0.15


def test_settings_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("LOOP_MAX_ITERATIONS", "8")
    s = Settings()
    assert s.loop_max_iterations == 8
