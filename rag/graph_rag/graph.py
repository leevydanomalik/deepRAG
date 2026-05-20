"""Graph RAG: extract entities from question → expand Neo4j subgraph →
fetch chunks the entities/relations point to → generate."""
from __future__ import annotations

import json
import re

from langgraph.graph import END, StateGraph

from rag.core.llm import get_chat_model
from rag.core.neo4j_store import Neo4jStore
from rag.core.pg_store import PgStore
from rag.core.prompts import ANSWER_PROMPT, GRAPH_RAG_ENTITY_EXTRACT_PROMPT
from rag.graph_rag.state import GraphRAGState


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _extract_entities(state: GraphRAGState) -> GraphRAGState:
    llm = get_chat_model(temperature=0.0)
    resp = llm.invoke(GRAPH_RAG_ENTITY_EXTRACT_PROMPT.format(question=state["question"]))
    parsed = _extract_json(getattr(resp, "content", ""))
    ents = [e for e in parsed.get("entities", []) if isinstance(e, str) and e.strip()]
    return {"entities": ents, "history": [{"node": "extract_entities", "entities": ents}]}


def _expand_subgraph(state: GraphRAGState) -> GraphRAGState:
    neo = Neo4jStore()
    nodes, edges = neo.expand_subgraph(state.get("entities", []), hops=2)
    neo.close()
    return {
        "subgraph_nodes": nodes,
        "subgraph_edges": edges,
        "history": [{"node": "expand_subgraph", "nodes": len(nodes), "edges": len(edges)}],
    }


def _fetch_chunks(state: GraphRAGState) -> GraphRAGState:
    chunk_ids: list[str] = []
    seen = set()
    for n in state.get("subgraph_nodes", []):
        for cid in n.get("source_chunks") or []:
            if cid not in seen:
                chunk_ids.append(cid); seen.add(cid)
    for e in state.get("subgraph_edges", []):
        for cid in e.get("source_chunks") or []:
            if cid not in seen:
                chunk_ids.append(cid); seen.add(cid)
    pg = PgStore()
    chunks = pg.fetch_by_ids(chunk_ids[:20])
    pg.close()
    return {"retrieved": chunks, "history": [{"node": "fetch_chunks", "n": len(chunks)}]}


def _generate(state: GraphRAGState) -> GraphRAGState:
    subgraph_summary = "\n".join(
        f"- ({e['src']}) -[{e['type']}]-> ({e['dst']}): {e.get('description', '')}"
        for e in state.get("subgraph_edges", [])
    )
    chunks_txt = "\n\n---\n\n".join(
        f"[{c['source']}#{c['chunk_index']}] {c['text']}" for c in state.get("retrieved", [])
    )
    ctx = f"Knowledge graph:\n{subgraph_summary}\n\nChunks:\n{chunks_txt}".strip() or "(no context)"
    prompt = ANSWER_PROMPT.format(context=ctx, question=state["question"])
    llm = get_chat_model(temperature=0.0)
    resp = llm.invoke(prompt)
    return {"answer": getattr(resp, "content", str(resp)), "history": [{"node": "generate"}]}


def build_graph():
    g = StateGraph(GraphRAGState)
    g.add_node("extract_entities", _extract_entities)
    g.add_node("expand_subgraph", _expand_subgraph)
    g.add_node("fetch_chunks", _fetch_chunks)
    g.add_node("generate", _generate)

    g.set_entry_point("extract_entities")
    g.add_edge("extract_entities", "expand_subgraph")
    g.add_edge("expand_subgraph", "fetch_chunks")
    g.add_edge("fetch_chunks", "generate")
    g.add_edge("generate", END)
    return g.compile()
