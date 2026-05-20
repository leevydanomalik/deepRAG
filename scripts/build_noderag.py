"""Build the NodeRAG heterogeneous-node index.

Pipeline (per chunk):
  1. Reuse existing entities + relations from the Graph RAG sqlite-graph DB.
  2. Extract semantic units (atomic claims) via LLM.
  3. Extract attributes (entity properties) via LLM.

Then globally:
  4. Embed every node (entity, relationship, semantic_unit, attribute) and
     upsert into pgvector (noderag_nodes table).
  5. Build a networkx graph from sqlite-graph edges + new node-type membership
     edges (semantic_unit ↔ entity, attribute ↔ entity).
  6. Louvain community detection.
  7. For each community: create a high_level_element node (community name + members)
     and an LLM-summarized high_level_overview node.
  8. Embed + upsert those too.

Usage:
    python -m scripts.build_noderag           # full run
    python -m scripts.build_noderag --skip-communities  # 4 node types only
    python -m scripts.build_noderag --limit 10   # cap for testing
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import networkx as nx
import typer

from rag.core.embeddings import embed_batch
from rag.core.graph_store import GraphStore
from rag.core.llm import get_chat_model
from rag.core.noderag_store import NodeRAGStore
from rag.core.pg_store import PgStore
from rag.core.prompts import (
    NODERAG_ATTRIBUTE_PROMPT,
    NODERAG_COMMUNITY_SUMMARY_PROMPT,
    NODERAG_SEMANTIC_UNIT_PROMPT,
)

app = typer.Typer()


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}:{h}"


# ---------- per-chunk LLM extractions ----------
def _extract_semantic_units(llm, text: str) -> list[dict]:
    resp = llm.invoke(NODERAG_SEMANTIC_UNIT_PROMPT.format(text=text[:4000]))
    parsed = _extract_json(getattr(resp, "content", ""))
    out = []
    for u in parsed.get("semantic_units", []) or []:
        claim = (u.get("claim") or "").strip()
        if not claim:
            continue
        ents = [e.strip() for e in (u.get("entities") or []) if e and e.strip()]
        out.append({"claim": claim, "entities": ents})
    return out


def _extract_attributes(llm, text: str) -> list[dict]:
    resp = llm.invoke(NODERAG_ATTRIBUTE_PROMPT.format(text=text[:4000]))
    parsed = _extract_json(getattr(resp, "content", ""))
    out = []
    for a in parsed.get("attributes", []) or []:
        ent = (a.get("entity") or "").strip()
        name = (a.get("name") or "").strip()
        val = a.get("value")
        if not ent or not name or val is None:
            continue
        out.append({"entity": ent, "name": name, "value": str(val).strip()})
    return out


# ---------- entity + relationship harvest from existing graph ----------
def _harvest_entities_and_relations(gs: GraphStore) -> tuple[list[dict], list[dict]]:
    """Pull all entities + relationships already in the sqlite-graph DB."""
    rows = gs.conn.execute(
        """
        SELECT i.name, i.display_name, n.properties
        FROM entity_name_idx i JOIN graph_nodes n ON n.id = i.node_id
        """
    ).fetchall()
    entities = []
    for name, display, raw in rows:
        p = json.loads(raw) if raw else {}
        entities.append(
            {
                "name": name,
                "display_name": display or name,
                "type": p.get("type", "Other"),
                "description": p.get("description", ""),
                "chunk_ids": list(p.get("source_chunks", []) or []),
            }
        )

    rel_rows = gs.conn.execute(
        """
        SELECT si.name, ti.name, e.edge_type, e.properties
        FROM graph_edges e
        JOIN entity_name_idx si ON si.node_id = e.source
        JOIN entity_name_idx ti ON ti.node_id = e.target
        """
    ).fetchall()
    relations = []
    for src, dst, etype, raw in rel_rows:
        p = json.loads(raw) if raw else {}
        relations.append(
            {
                "src": src,
                "dst": dst,
                "type": etype,
                "description": p.get("description", ""),
                "chunk_ids": list(p.get("source_chunks", []) or []),
            }
        )
    return entities, relations


# ---------- bulk embed + upsert helpers ----------
BATCH = 64


def _embed_and_upsert(
    store: NodeRAGStore,
    items: list[dict],
    fmt_id,
    fmt_content,
    node_type: str,
    fmt_metadata=None,
    fmt_chunk_ids=None,
) -> int:
    if not items:
        return 0
    contents = [fmt_content(it) for it in items]
    ids = [fmt_id(it) for it in items]
    metas = [fmt_metadata(it) if fmt_metadata else {} for it in items]
    chunks = [fmt_chunk_ids(it) if fmt_chunk_ids else [] for it in items]
    embedded = 0
    for i in range(0, len(items), BATCH):
        slc = slice(i, i + BATCH)
        vecs = embed_batch(contents[slc])
        for nid, c, e, m, ch in zip(ids[slc], contents[slc], vecs, metas[slc], chunks[slc]):
            store.upsert_node(
                node_id=nid,
                node_type=node_type,
                content=c,
                embedding=e,
                chunk_ids=ch,
                metadata=m,
            )
            embedded += 1
    return embedded


# ---------- main pipeline ----------
def run_build_noderag(limit: int | None = None, skip_communities: bool = False) -> dict:
    pg = PgStore()
    gs = GraphStore()
    nr = NodeRAGStore()
    nr.init_schema()
    llm = get_chat_model(temperature=0.0)

    # 1. read chunks
    sql = "SELECT id, source, chunk_index, text FROM rag_chunks ORDER BY source, chunk_index"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with pg.conn.cursor() as cur:
        cur.execute(sql)
        chunks = cur.fetchall()
    typer.echo(f"[1/5] {len(chunks)} chunks loaded.")

    # 2. extract semantic units + attributes per chunk
    all_units: list[dict] = []  # {claim, entities, chunk_id}
    all_attrs: list[dict] = []  # {entity, name, value, chunk_id}
    for i, (cid, _src, _idx, text) in enumerate(chunks, start=1):
        units = _extract_semantic_units(llm, text)
        attrs = _extract_attributes(llm, text)
        for u in units:
            u["chunk_id"] = cid
            all_units.append(u)
        for a in attrs:
            a["chunk_id"] = cid
            all_attrs.append(a)
        if i % 10 == 0 or i == len(chunks):
            typer.echo(f"  per-chunk extract {i}/{len(chunks)} (units={len(all_units)} attrs={len(all_attrs)})")
    typer.echo(f"[2/5] extracted {len(all_units)} semantic_units, {len(all_attrs)} attributes")

    # 3. harvest entities + relationships from existing graph_store
    entities, relations = _harvest_entities_and_relations(gs)
    typer.echo(f"[3/5] harvested {len(entities)} entities, {len(relations)} relations")

    # 4. embed + upsert all four base node types
    n_ent = _embed_and_upsert(
        nr, entities,
        fmt_id=lambda e: _stable_id("ent", e["name"]),
        fmt_content=lambda e: f"{e['display_name']} ({e['type']}): {e['description']}".strip(),
        node_type="entity",
        fmt_metadata=lambda e: {"name": e["name"], "display_name": e["display_name"], "entity_type": e["type"]},
        fmt_chunk_ids=lambda e: e["chunk_ids"],
    )
    typer.echo(f"  upserted {n_ent} entity nodes")

    n_rel = _embed_and_upsert(
        nr, relations,
        fmt_id=lambda r: _stable_id("rel", r["src"], r["type"], r["dst"]),
        fmt_content=lambda r: f"{r['src']} -[{r['type']}]-> {r['dst']}: {r['description']}".strip(),
        node_type="relationship",
        fmt_metadata=lambda r: {"src": r["src"], "dst": r["dst"], "type": r["type"]},
        fmt_chunk_ids=lambda r: r["chunk_ids"],
    )
    typer.echo(f"  upserted {n_rel} relationship nodes")

    n_unit = _embed_and_upsert(
        nr, all_units,
        fmt_id=lambda u: _stable_id("unit", u["chunk_id"], u["claim"]),
        fmt_content=lambda u: u["claim"],
        node_type="semantic_unit",
        fmt_metadata=lambda u: {"entities": u["entities"]},
        fmt_chunk_ids=lambda u: [u["chunk_id"]],
    )
    typer.echo(f"  upserted {n_unit} semantic_unit nodes")

    n_attr = _embed_and_upsert(
        nr, all_attrs,
        fmt_id=lambda a: _stable_id("attr", a["entity"], a["name"], a["value"]),
        fmt_content=lambda a: f"{a['entity']}.{a['name']} = {a['value']}",
        node_type="attribute",
        fmt_metadata=lambda a: {"entity": a["entity"], "name": a["name"], "value": a["value"]},
        fmt_chunk_ids=lambda a: [a["chunk_id"]],
    )
    typer.echo(f"  upserted {n_attr} attribute nodes")

    out = {
        "entity": n_ent,
        "relationship": n_rel,
        "semantic_unit": n_unit,
        "attribute": n_attr,
        "high_level_element": 0,
        "high_level_overview": 0,
    }

    if skip_communities:
        typer.echo("[5/5] skipping communities (--skip-communities)")
        pg.close(); gs.close(); nr.close()
        return out

    # 5. build a networkx graph for community detection
    typer.echo("[4/5] building networkx graph for community detection…")
    g = nx.Graph()
    # entity-entity edges from relations
    for r in relations:
        g.add_edge(_stable_id("ent", r["src"]), _stable_id("ent", r["dst"]), weight=1.0)
    # semantic_unit ↔ entity edges
    name_to_id = {e["name"]: _stable_id("ent", e["name"]) for e in entities}
    for u in all_units:
        u_id = _stable_id("unit", u["chunk_id"], u["claim"])
        for ent_name in u["entities"]:
            ent_lower = ent_name.strip().lower()
            if ent_lower in name_to_id:
                g.add_edge(u_id, name_to_id[ent_lower], weight=0.7)
    # attribute ↔ entity edges
    for a in all_attrs:
        a_id = _stable_id("attr", a["entity"], a["name"], a["value"])
        ent_lower = a["entity"].strip().lower()
        if ent_lower in name_to_id:
            g.add_edge(a_id, name_to_id[ent_lower], weight=0.5)

    typer.echo(f"  graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    if g.number_of_nodes() < 4:
        typer.echo("  graph too small for community detection; skipping high-level nodes")
        pg.close(); gs.close(); nr.close()
        return out

    # 6. Louvain communities
    try:
        import community as community_louvain  # python-louvain
    except ImportError as e:
        typer.echo(f"  python-louvain not installed: {e}; skipping communities")
        pg.close(); gs.close(); nr.close()
        return out

    partition = community_louvain.best_partition(g)
    communities: dict[int, list[str]] = {}
    for node_id, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node_id)
    typer.echo(f"  found {len(communities)} communities")

    # 7. high_level_element nodes (community-level concepts)
    id_to_meta: dict[str, dict] = {}
    for e in entities:
        id_to_meta[_stable_id("ent", e["name"])] = {"type": "entity", "display": e["display_name"]}
    for u in all_units:
        id_to_meta[_stable_id("unit", u["chunk_id"], u["claim"])] = {"type": "semantic_unit", "display": u["claim"][:80]}
    for a in all_attrs:
        id_to_meta[_stable_id("attr", a["entity"], a["name"], a["value"])] = {
            "type": "attribute",
            "display": f"{a['entity']}.{a['name']} = {a['value']}",
        }

    hle_items: list[dict] = []
    hlo_items: list[dict] = []
    for comm_id, members in communities.items():
        if len(members) < 3:
            continue
        labels = [id_to_meta.get(m, {}).get("display", m) for m in members[:20]]
        members_text = "\n".join(f"- {l}" for l in labels)
        community_label = f"Community {comm_id} ({len(members)} members)"

        # collect chunk ids referenced by community members
        chunk_ids: list[str] = []
        seen = set()
        for m in members:
            # find this node's chunk_ids in pgvector (it must be already upserted)
            row = nr.conn.execute(
                "SELECT chunk_ids FROM noderag_nodes WHERE id = %s", (m,)
            ).fetchone()
            if row and row[0]:
                for cid in row[0]:
                    if cid not in seen:
                        chunk_ids.append(cid); seen.add(cid)

        hle_items.append({
            "comm_id": comm_id,
            "label": community_label,
            "members_text": members_text,
            "chunk_ids": chunk_ids,
            "n_members": len(members),
        })

    n_hle = _embed_and_upsert(
        nr, hle_items,
        fmt_id=lambda h: _stable_id("hle", str(h["comm_id"])),
        fmt_content=lambda h: f"{h['label']}\nMembers:\n{h['members_text']}",
        node_type="high_level_element",
        fmt_metadata=lambda h: {"comm_id": h["comm_id"], "n_members": h["n_members"]},
        fmt_chunk_ids=lambda h: h["chunk_ids"],
    )
    typer.echo(f"  upserted {n_hle} high_level_element nodes")
    out["high_level_element"] = n_hle

    # 8. high_level_overview — LLM summary per community
    for h in hle_items:
        resp = llm.invoke(NODERAG_COMMUNITY_SUMMARY_PROMPT.format(members=h["members_text"]))
        summary = getattr(resp, "content", "").strip()
        if not summary:
            continue
        hlo_items.append({
            "comm_id": h["comm_id"],
            "summary": summary,
            "chunk_ids": h["chunk_ids"],
        })

    n_hlo = _embed_and_upsert(
        nr, hlo_items,
        fmt_id=lambda h: _stable_id("hlo", str(h["comm_id"])),
        fmt_content=lambda h: h["summary"],
        node_type="high_level_overview",
        fmt_metadata=lambda h: {"comm_id": h["comm_id"]},
        fmt_chunk_ids=lambda h: h["chunk_ids"],
    )
    typer.echo(f"  upserted {n_hlo} high_level_overview nodes")
    out["high_level_overview"] = n_hlo

    pg.close(); gs.close(); nr.close()
    typer.echo(f"[5/5] Done. {out}")
    return out


@app.command()
def main(
    limit: int = typer.Option(None, help="Cap number of chunks (for testing)"),
    skip_communities: bool = typer.Option(False, "--skip-communities", help="Skip Louvain + high-level summaries"),
):
    run_build_noderag(limit=limit, skip_communities=skip_communities)


if __name__ == "__main__":
    app()
