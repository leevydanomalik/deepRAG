from rag.core.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("PG_DSN", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")

    s = Settings()
    assert s.llm_api_key.get_secret_value() == "sk-test"
    assert s.llm_model == "deepseek-chat"
    assert s.embedding_provider == "local"
    assert s.embedding_dim == 384
    assert s.loop_max_iterations == 4
    assert s.loop_convergence_threshold == 0.15


def test_settings_overrides(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("LOOP_MAX_ITERATIONS", "8")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    s = Settings()
    assert s.loop_max_iterations == 8
    assert s.embedding_provider == "openai"
    assert s.embedding_dim == 1536
