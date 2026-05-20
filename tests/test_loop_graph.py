import json
from unittest.mock import MagicMock, patch

from rag.loop.graph import build_graph


def _env(monkeypatch):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "PG_DSN"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_loop_converges_on_first_pass(monkeypatch):
    _env(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({
            "task_type": "factual", "entities": [], "constraints": [],
            "sub_goals": [], "prompt_template": "stuff"
        })),
        MagicMock(content="this is the answer."),
        MagicMock(content=json.dumps({"support_scores": [1.0]})),
    ]

    fake_emb = MagicMock()
    fake_emb.embed_query.return_value = [1.0, 0.0]

    fake_pg = MagicMock()
    fake_pg.similarity_search.return_value = [
        {"id": "c1", "source": "a.md", "chunk_index": 0, "text": "alpha", "metadata": {}, "score": 0.9}
    ]
    fake_pg.fetch_by_ids.return_value = []
    fake_gs = MagicMock()
    fake_gs.expand_subgraph.return_value = ([], [])

    with patch("rag.loop.agents.get_chat_model", return_value=fake_llm), \
         patch("rag.loop.agents.embed_text", return_value=[1.0, 0.0]), \
         patch("rag.loop.agents.get_embeddings", return_value=fake_emb), \
         patch("rag.loop.agents.PgStore", return_value=fake_pg), \
         patch("rag.loop.agents.GraphStore", return_value=fake_gs):
        app = build_graph()
        out = app.invoke({"question": "what is alpha?", "iteration": 0})

    assert out["answer"] == "this is the answer."
    assert out["converged"] is True
