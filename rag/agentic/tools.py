"""Tools the Agentic RAG agent can call."""
from __future__ import annotations

from langchain_core.tools import tool

from rag.core.embeddings import embed_text
from rag.core.graph_store import GraphStore
from rag.core.pg_store import PgStore


@tool
def vector_search(query: str, k: int = 5) -> str:
    """Search the corpus by semantic similarity. Returns top-k chunks as text."""
    store = PgStore()
    hits = store.similarity_search(embed_text(query), k=k)
    store.close()
    if not hits:
        return "No results."
    return "\n\n---\n\n".join(
        f"[{h['source']}#{h['chunk_index']}] {h['text']}" for h in hits
    )


@tool
def kg_lookup(entity_name: str) -> str:
    """Look up an entity in the knowledge graph; returns description + 1-hop neighbors."""
    gs = GraphStore()
    ent = gs.get_entity(entity_name)
    if not ent:
        gs.close()
        return f"Entity '{entity_name}' not found."
    rels = gs.find_relations([entity_name])
    gs.close()
    lines = [f"Entity: {ent.get('display_name', entity_name)} ({ent.get('type', '?')})",
             f"Description: {ent.get('description', '')}"]
    if rels:
        lines.append("Relations:")
        for r in rels[:10]:
            lines.append(f"  - {r['src']} -[{r['type']}]-> {r['dst']}: {r.get('description', '')}")
    return "\n".join(lines)


TOOLS = [vector_search, kg_lookup]
