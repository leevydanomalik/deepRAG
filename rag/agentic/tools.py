"""Tools the Agentic RAG agent can call."""
from __future__ import annotations

from langchain_core.tools import tool

from rag.core.embeddings import embed_text
from rag.core.neo4j_store import Neo4jStore
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
    neo = Neo4jStore()
    ent = neo.get_entity(entity_name)
    if not ent:
        neo.close()
        return f"Entity '{entity_name}' not found."
    rels = neo.find_relations([entity_name])
    neo.close()
    lines = [f"Entity: {ent.get('display_name', entity_name)} ({ent.get('type', '?')})",
             f"Description: {ent.get('description', '')}"]
    if rels:
        lines.append("Relations:")
        for r in rels[:10]:
            lines.append(f"  - {r['src']} -[{r['type']}]-> {r['dst']}: {r.get('description', '')}")
    return "\n".join(lines)


TOOLS = [vector_search, kg_lookup]
