"""Plan-Do-Check-Act agents for Loop RAG."""
from __future__ import annotations

import json
import math
import re
from typing import Any

from rag.core.config import get_settings
from rag.core.embeddings import embed_text, get_embeddings
from rag.core.llm import get_chat_model
from rag.core.neo4j_store import Neo4jStore
from rag.core.pg_store import PgStore
from rag.core.prompts import (
    ANSWER_PROMPT,
    LOOP_ACT_PROMPT,
    LOOP_CHECK_PROMPT,
    LOOP_PLAN_PROMPT,
)
from rag.loop.state import LoopState


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def plan_node(state: LoopState) -> LoopState:
    question = state.get("question") or state.get("user_input") or ""
    llm = get_chat_model(temperature=0.0)
    parsed = _extract_json(getattr(llm.invoke(LOOP_PLAN_PROMPT.format(question=question)), "content", ""))
    intent = {
        "task_type": parsed.get("task_type", "factual"),
        "entities": parsed.get("entities", []) or [],
        "constraints": parsed.get("constraints", []) or [],
        "sub_goals": parsed.get("sub_goals", []) or [],
    }
    prompt_config = {"template": parsed.get("prompt_template", "stuff")}
    return {
        "question": question,
        "user_input": question,
        "intent": intent,
        "prompt_config": prompt_config,
        "iteration": state.get("iteration", 0),
        "converged": False,
        "history": [{"node": "plan", "intent": intent, "iter": state.get("iteration", 0)}],
    }


def make_do_node(pg_store=None, neo_store=None):
    """Factory so tests can inject fakes."""
    def _do(state: LoopState) -> LoopState:
        pg = pg_store or PgStore()
        neo = neo_store or Neo4jStore()

        q = state["question"]
        q_emb = embed_text(q)
        hits = pg.similarity_search(q_emb, k=8)

        entities = state.get("intent", {}).get("entities", []) or []
        graph_chunks: list[dict[str, Any]] = []
        if entities:
            try:
                nodes, edges = neo.expand_subgraph(entities, hops=2)
                ids: list[str] = []
                seen = set()
                for n in nodes:
                    for cid in n.get("source_chunks") or []:
                        if cid not in seen:
                            ids.append(cid); seen.add(cid)
                for e in edges:
                    for cid in e.get("source_chunks") or []:
                        if cid not in seen:
                            ids.append(cid); seen.add(cid)
                graph_chunks = pg.fetch_by_ids(ids[:8])
            except Exception:
                graph_chunks = []

        merged: dict[str, dict] = {}
        for h in hits:
            merged[h["id"]] = {**h, "score": h.get("score", 0.0)}
        for c in graph_chunks:
            if c["id"] not in merged:
                merged[c["id"]] = {**c, "score": 0.5}
        evidence = list(merged.values())[:8]

        ctx = "\n\n---\n\n".join(
            f"[{e['source']}#{e['chunk_index']}] {e['text']}" for e in evidence
        )
        prompt = ANSWER_PROMPT.format(context=ctx or "(no context)", question=q)
        llm = get_chat_model(temperature=0.0)
        resp = llm.invoke(prompt)

        if pg_store is None:
            pg.close()
        if neo_store is None:
            neo.close()

        return {
            "evidence": evidence,
            "answer": getattr(resp, "content", ""),
            "history": [{"node": "do", "evidence": len(evidence)}],
        }
    return _do


def check_node(state: LoopState) -> LoopState:
    s = get_settings()
    q = state["question"]
    a = state.get("answer", "")
    evidence = state.get("evidence", [])

    emb = get_embeddings()
    e_q = emb.embed_query(q)
    e_a = emb.embed_query(a) if a else [0.0] * len(e_q)
    align = 1.0 - cosine(e_q, e_a)

    ev_lines = "\n".join(f"{i+1}. {e['text']}" for i, e in enumerate(evidence))
    faith = 1.0
    if evidence:
        llm = get_chat_model(temperature=0.0)
        parsed = _extract_json(getattr(
            llm.invoke(LOOP_CHECK_PROMPT.format(question=q, answer=a, evidence=ev_lines)),
            "content", ""))
        scores = [float(x) for x in parsed.get("support_scores", []) if isinstance(x, (int, float))]
        if scores:
            faith = 1.0 - (sum(scores) / len(scores))

    constraint = 0.0 if a and len(a) >= 20 else 0.5

    total = 0.5 * align + 0.4 * faith + 0.1 * constraint
    dominant = max(
        [("align", align), ("faith", faith), ("constraint", constraint)],
        key=lambda kv: kv[1],
    )[0]

    iteration = state.get("iteration", 0)
    converged = total < s.loop_convergence_threshold or iteration + 1 >= s.loop_max_iterations
    dev = {"align": align, "faith": faith, "constraint": constraint,
           "total": total, "dominant": dominant, "converged": converged}

    return {
        "deviation": dev,
        "converged": converged,
        "history": [{"node": "check", "deviation": dev, "iter": iteration}],
    }


def act_node(state: LoopState) -> LoopState:
    llm = get_chat_model(temperature=0.0)
    dom = state.get("deviation", {}).get("dominant", "align")
    parsed = _extract_json(getattr(
        llm.invoke(LOOP_ACT_PROMPT.format(
            dominant=dom, question=state["question"], answer=state.get("answer", "")
        )),
        "content", ""))
    new_q = parsed.get("rewritten_query") or state["question"]
    new_template = parsed.get("prompt_template", state.get("prompt_config", {}).get("template", "stuff"))
    return {
        "question": new_q,
        "user_input": new_q,
        "prompt_config": {"template": new_template},
        "iteration": state.get("iteration", 0) + 1,
        "history": [{"node": "act", "rewritten_query": new_q, "template": new_template}],
    }


def should_continue(state: LoopState) -> str:
    return "end" if state.get("converged") else "continue"
