from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_patterns():
    client = TestClient(app)
    with patch("app.api.list_patterns", return_value=["naive", "agentic", "graph", "loop"]):
        r = client.get("/patterns")
    assert r.status_code == 200
    assert r.json() == ["naive", "agentic", "graph", "loop"]


def test_rag_endpoint():
    client = TestClient(app)
    with patch("app.api.run", return_value={"answer": "yes", "trace": [], "raw": {}}):
        r = client.post("/rag/naive", json={"question": "is it?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "yes"
    assert "latency_ms" in body
