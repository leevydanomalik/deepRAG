"""Walk pg_chunks → ask LLM for entities/relations → MERGE into Neo4j."""
from __future__ import annotations

import json
import re

import typer

from rag.core.llm import get_chat_model
from rag.core.graph_store import GraphStore
from rag.core.pg_store import PgStore
from rag.core.prompts import KG_EXTRACTION_PROMPT

app = typer.Typer()


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def extract_kg_from_chunk(llm, text: str) -> dict:
    prompt = KG_EXTRACTION_PROMPT.format(text=text[:4000])
    resp = llm.invoke(prompt)
    payload = _extract_json(getattr(resp, "content", ""))
    return {
        "entities": payload.get("entities", []) or [],
        "relations": payload.get("relations", []) or [],
    }


def run_build_kg(limit: int | None = None) -> tuple[int, int]:
    pg = PgStore()
    gs = GraphStore()
    gs.init_schema()
    llm = get_chat_model(temperature=0.0)

    sql = "SELECT id, source, chunk_index, text, metadata FROM rag_chunks ORDER BY source, chunk_index"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    entities_added = 0
    relations_added = 0

    with pg.conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    typer.echo(f"Extracting KG from {len(rows)} chunks…")
    for row in rows:
        chunk_id = row[0]
        text = row[3]
        kg = extract_kg_from_chunk(llm, text)

        valid_names = set()
        for e in kg["entities"]:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            gs.merge_entity(
                name=name,
                type_=e.get("type", "Other"),
                description=e.get("description", ""),
                chunk_ids=[chunk_id],
            )
            valid_names.add(name.lower())
            entities_added += 1

        for r in kg["relations"]:
            src = (r.get("src") or "").strip()
            dst = (r.get("dst") or "").strip()
            if not src or not dst:
                continue
            if src.lower() not in valid_names or dst.lower() not in valid_names:
                continue
            gs.merge_relation(
                src=src,
                dst=dst,
                type_=r.get("type", "RELATED_TO"),
                description=r.get("description", ""),
                chunk_ids=[chunk_id],
            )
            relations_added += 1

    pg.close()
    gs.close()
    typer.echo(f"Done. entities={entities_added} relations={relations_added}")
    return entities_added, relations_added


@app.command()
def main(limit: int = typer.Option(None, help="Cap number of chunks to process")):
    run_build_kg(limit=limit)


if __name__ == "__main__":
    app()
