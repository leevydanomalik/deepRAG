"""Naive RAG: retrieve top-k → stuff prompt → generate."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag.core.embeddings import embed_text
from rag.core.llm import get_chat_model
from rag.core.pg_store import PgStore
from rag.core.prompts import ANSWER_PROMPT
from rag.naive.state import NaiveState


def _retrieve_node(state: NaiveState) -> NaiveState:
    store = PgStore()
    q_emb = embed_text(state["question"])
    hits = store.similarity_search(q_emb, k=5)
    store.close()
    return {
        "retrieved": hits,
        "history": [{"node": "retrieve", "hits": [h["id"] for h in hits]}],
    }


def _generate_node(state: NaiveState) -> NaiveState:
    ctx = "\n\n---\n\n".join(
        f"[{h['source']}#{h['chunk_index']}] {h['text']}" for h in state.get("retrieved", [])
    )
    prompt = ANSWER_PROMPT.format(context=ctx or "(no context)", question=state["question"])
    llm = get_chat_model(temperature=0.0)
    resp = llm.invoke(prompt)
    return {
        "answer": getattr(resp, "content", str(resp)),
        "history": [{"node": "generate"}],
    }


def build_graph():
    g = StateGraph(NaiveState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("generate", _generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
