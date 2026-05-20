"""NodeRAG: heterogeneous-node RAG with shallow (HNSW) + deep (PPR) retrieval.

Pipeline:
    embed_query → vector_seed → ppr_propagate → fetch_chunks → generate

vector_seed runs HNSW cosine search across noderag_nodes for top-K seeds
across all 6 node types. ppr_propagate runs Personalized PageRank seeded by
those nodes over the heterogeneous graph (entity-entity from sqlite-graph,
plus entity↔semantic_unit, entity↔attribute, community↔members membership
edges). fetch_chunks pulls source chunks for the top-N PPR-ranked nodes.
"""
from __future__ import annotations

import functools
import json

import networkx as nx
from langgraph.graph import END, StateGraph

from rag.core.config import get_settings
from rag.core.embeddings import embed_text
from rag.core.graph_store import GraphStore
from rag.core.llm import get_chat_model
from rag.core.noderag_store import NodeRAGStore
from rag.core.pg_store import PgStore
from rag.core.prompts import ANSWER_PROMPT
from rag.noderag.state import NodeRAGState


def _stable_id(prefix: str, *parts: str) -> str:
    import hashlib
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}:{h}"


@functools.lru_cache(maxsize=1)
def _build_ppr_graph() -> nx.Graph:
    """Construct the heterogeneous graph for PPR. Built once, cached.

    Nodes are noderag_node IDs. Edges:
      - entity-entity from sqlite-graph relations (weight 1.0)
      - semantic_unit ↔ entity from semantic_unit metadata.entities (weight 0.7)
      - attribute ↔ entity from attribute metadata.entity (weight 0.5)
      - high_level_element/overview ↔ members are NOT added as edges here;
        PPR can rank them via their direct membership through shared chunk_ids
        instead — keeps the graph sparser.
    """
    nr = NodeRAGStore()
    gs = GraphStore()
    g = nx.Graph()

    # 1. all nodes
    for nid, _ntype in nr.all_nodes_minimal():
        g.add_node(nid)

    # 2. entity-entity edges from sqlite-graph relations
    rel_rows = gs.conn.execute(
        """
        SELECT si.name, ti.name
        FROM graph_edges e
        JOIN entity_name_idx si ON si.node_id = e.source
        JOIN entity_name_idx ti ON ti.node_id = e.target
        """
    ).fetchall()
    for src, dst in rel_rows:
        sid = _stable_id("ent", src)
        did = _stable_id("ent", dst)
        if g.has_node(sid) and g.has_node(did):
            g.add_edge(sid, did, weight=1.0)
    gs.close()

    # 3. semantic_unit ↔ entity, attribute ↔ entity
    with nr.conn.cursor() as cur:
        cur.execute(
            "SELECT id, node_type, metadata FROM noderag_nodes WHERE node_type IN ('semantic_unit', 'attribute');"
        )
        rows = cur.fetchall()
    for nid, ntype, raw_meta in rows:
        meta = raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta or "{}")
        if ntype == "semantic_unit":
            for ent_name in meta.get("entities", []) or []:
                ent_id = _stable_id("ent", ent_name.strip().lower())
                if g.has_node(ent_id):
                    g.add_edge(nid, ent_id, weight=0.7)
        elif ntype == "attribute":
            ent_name = meta.get("entity")
            if ent_name:
                ent_id = _stable_id("ent", ent_name.strip().lower())
                if g.has_node(ent_id):
                    g.add_edge(nid, ent_id, weight=0.5)
    nr.close()
    return g


def _embed_query_node(state: NodeRAGState) -> NodeRAGState:
    emb = embed_text(state["question"])
    return {
        "q_embedding": emb,
        "history": [{"node": "embed_query", "dim": len(emb)}],
    }


def _vector_seed_node(state: NodeRAGState) -> NodeRAGState:
    s = get_settings()
    nr = NodeRAGStore()
    q_emb = state.get("q_embedding") or embed_text(state["question"])
    seeds = nr.vector_search(q_emb, k=s.noderag_top_k_seeds)
    nr.close()
    return {
        "seed_nodes": seeds,
        "history": [
            {
                "node": "vector_seed",
                "k": len(seeds),
                "types": {t: sum(1 for sd in seeds if sd["node_type"] == t) for t in {sd["node_type"] for sd in seeds}},
            }
        ],
    }


def _ppr_propagate_node(state: NodeRAGState) -> NodeRAGState:
    s = get_settings()
    seeds = state.get("seed_nodes", [])
    if not seeds:
        return {"ranked_nodes": [], "ppr_scores": {}, "history": [{"node": "ppr_propagate", "expanded": 0}]}

    g = _build_ppr_graph()
    # personalization vector: 1.0 weight on seeds (proportional to seed cosine score)
    personalization = {}
    for s_node in seeds:
        nid = s_node["id"]
        if g.has_node(nid):
            personalization[nid] = max(float(s_node.get("score", 0.5)), 0.0)
    # guard: if none of the seeds are in the graph, fall back to seeds themselves
    if not personalization:
        ranked_ids = [sd["id"] for sd in seeds][: s.noderag_ppr_top_n]
        ppr_scores = {sid: float(sd.get("score", 0.0)) for sid, sd in zip(ranked_ids, seeds)}
        nr = NodeRAGStore()
        full = nr.fetch_by_ids(ranked_ids)
        nr.close()
        full_by_id = {n["id"]: n for n in full}
        ranked = [full_by_id[i] for i in ranked_ids if i in full_by_id]
        for r in ranked:
            r["ppr_score"] = ppr_scores.get(r["id"], 0.0)
        return {
            "ranked_nodes": ranked,
            "ppr_scores": ppr_scores,
            "history": [{"node": "ppr_propagate", "expanded": len(ranked), "fallback": True}],
        }

    pr = nx.pagerank(g, alpha=1 - s.noderag_ppr_alpha, personalization=personalization, max_iter=100)
    top_ids = [nid for nid, _ in sorted(pr.items(), key=lambda kv: -kv[1])[: s.noderag_ppr_top_n]]

    nr = NodeRAGStore()
    full = nr.fetch_by_ids(top_ids)
    nr.close()
    by_id = {n["id"]: n for n in full}
    ranked = []
    for nid in top_ids:
        if nid in by_id:
            row = by_id[nid]
            row["ppr_score"] = pr[nid]
            ranked.append(row)
    return {
        "ranked_nodes": ranked,
        "ppr_scores": {nid: float(pr[nid]) for nid in top_ids},
        "history": [{"node": "ppr_propagate", "expanded": len(ranked), "graph_nodes": g.number_of_nodes()}],
    }


def _fetch_chunks_node(state: NodeRAGState) -> NodeRAGState:
    s = get_settings()
    chunk_ids: list[str] = []
    seen: set[str] = set()
    for n in state.get("ranked_nodes", []):
        for cid in n.get("chunk_ids") or []:
            if cid not in seen:
                chunk_ids.append(cid); seen.add(cid)
            if len(chunk_ids) >= s.noderag_chunk_top_k:
                break
        if len(chunk_ids) >= s.noderag_chunk_top_k:
            break

    pg = PgStore()
    chunks = pg.fetch_by_ids(chunk_ids)
    pg.close()
    return {
        "retrieved": chunks,
        "history": [{"node": "fetch_chunks", "n": len(chunks)}],
    }


def _generate_node(state: NodeRAGState) -> NodeRAGState:
    # context = ranked node descriptions + actual chunk text
    node_lines = []
    for n in state.get("ranked_nodes", [])[:15]:
        node_lines.append(f"[{n['node_type']} score={n.get('ppr_score', 0):.4f}] {n['content']}")
    chunk_lines = [
        f"[{c['source']}#{c['chunk_index']}] {c['text']}"
        for c in state.get("retrieved", [])
    ]
    ctx_parts = []
    if node_lines:
        ctx_parts.append("Knowledge graph nodes:\n" + "\n".join(node_lines))
    if chunk_lines:
        ctx_parts.append("Source chunks:\n" + "\n\n---\n\n".join(chunk_lines))
    ctx = "\n\n".join(ctx_parts) or "(no context)"
    prompt = ANSWER_PROMPT.format(context=ctx, question=state["question"])
    llm = get_chat_model(temperature=0.0)
    resp = llm.invoke(prompt)
    return {
        "answer": getattr(resp, "content", str(resp)),
        "history": [{"node": "generate"}],
    }


def build_graph():
    g = StateGraph(NodeRAGState)
    g.add_node("embed_query", _embed_query_node)
    g.add_node("vector_seed", _vector_seed_node)
    g.add_node("ppr_propagate", _ppr_propagate_node)
    g.add_node("fetch_chunks", _fetch_chunks_node)
    g.add_node("generate", _generate_node)

    g.set_entry_point("embed_query")
    g.add_edge("embed_query", "vector_seed")
    g.add_edge("vector_seed", "ppr_propagate")
    g.add_edge("ppr_propagate", "fetch_chunks")
    g.add_edge("fetch_chunks", "generate")
    g.add_edge("generate", END)
    return g.compile()
