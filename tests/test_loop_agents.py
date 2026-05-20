import json
import math
from unittest.mock import MagicMock, patch

import pytest

from rag.loop.agents import (
    act_node,
    check_node,
    cosine,
    make_do_node,
    plan_node,
    should_continue,
)


def _env(monkeypatch):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "PG_DSN"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_cosine_zero_vec():
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_identity():
    assert math.isclose(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)


def test_plan_node_parses_json(monkeypatch):
    _env(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "task_type": "factual",
        "entities": ["LangGraph"],
        "constraints": [],
        "sub_goals": ["define"],
        "prompt_template": "stuff",
    }))
    with patch("rag.loop.agents.get_chat_model", return_value=fake_llm):
        out = plan_node({"question": "what is LangGraph?"})
    assert out["intent"]["task_type"] == "factual"
    assert out["prompt_config"]["template"] == "stuff"
    assert out["iteration"] == 0


def test_should_continue():
    s = {"converged": False, "iteration": 1}
    assert should_continue(s) == "continue"
    assert should_continue({"converged": True, "iteration": 1}) == "end"


def test_check_node_computes_deviation(monkeypatch):
    _env(monkeypatch)
    fake_emb = MagicMock()
    fake_emb.embed_query.side_effect = [[1.0, 0.0], [1.0, 0.0]]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({"support_scores": [1.0, 1.0]}))
    with patch("rag.loop.agents.get_embeddings", return_value=fake_emb), \
         patch("rag.loop.agents.get_chat_model", return_value=fake_llm):
        out = check_node({
            "question": "q",
            "answer": "a",
            "evidence": [{"id": "1", "text": "x", "score": 0.9}, {"id": "2", "text": "y", "score": 0.9}],
            "iteration": 0,
            "prompt_config": {"template": "stuff"},
        })
    dev = out["deviation"]
    assert 0.0 <= dev["align"] <= 1.0
    assert 0.0 <= dev["faith"] <= 1.0
    assert dev["converged"] is True or dev["converged"] is False


def test_act_node_increments_iteration(monkeypatch):
    _env(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "rewritten_query": "new q", "prompt_template": "stepwise",
    }))
    with patch("rag.loop.agents.get_chat_model", return_value=fake_llm):
        out = act_node({
            "question": "old",
            "answer": "bad",
            "deviation": {"dominant": "align"},
            "iteration": 1,
        })
    assert out["question"] == "new q"
    assert out["iteration"] == 2
    assert out["prompt_config"]["template"] == "stepwise"
