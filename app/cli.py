"""Typer CLI: ask a pattern a question, run ingestion, build KG, reset."""
from __future__ import annotations

import json

import typer

from rag.registry import list_patterns, run

app = typer.Typer(help="DIP RAG CLI")


@app.command()
def patterns():
    """List available RAG patterns."""
    for p in list_patterns():
        typer.echo(p)


@app.command()
def ask(pattern: str, question: str, show_trace: bool = typer.Option(False, "--trace")):
    """Ask QUESTION against the chosen PATTERN."""
    result = run(pattern, question)
    typer.echo("\n=== ANSWER ===\n")
    typer.echo(result["answer"])
    if show_trace:
        typer.echo("\n=== TRACE ===\n")
        typer.echo(json.dumps(result["trace"], indent=2, default=str))


@app.command()
def compare(
    question: str,
    patterns: str = typer.Option(
        "naive,agentic,graph,loop,noderag",
        help="Comma-separated pattern names to run",
    ),
):
    """Run QUESTION through multiple patterns and print a side-by-side summary."""
    import time
    names = [p.strip() for p in patterns.split(",") if p.strip()]
    results = []
    for p in names:
        if p not in list_patterns():
            typer.echo(f"  skip: unknown pattern {p!r}")
            continue
        typer.echo(f"▶ running {p}…")
        t0 = time.perf_counter()
        try:
            r = run(p, question)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            results.append((p, r["answer"], r["trace"], dt_ms, None))
        except Exception as e:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            results.append((p, None, [], dt_ms, str(e)))

    typer.echo("\n" + "═" * 80)
    typer.echo(f"QUESTION: {question}")
    typer.echo("═" * 80)
    for p, ans, trace, dt_ms, err in results:
        typer.echo(f"\n▼ {p.upper()}   ({dt_ms} ms, {len(trace)} trace steps)")
        typer.echo("─" * 80)
        if err is not None:
            typer.echo(f"  ERROR: {err}")
        else:
            typer.echo(ans)
            typer.echo("\n  trace nodes: " + " → ".join(h.get("node", "?") for h in trace))


@app.command()
def ingest(data_dir: str = typer.Option(None)):
    """Chunk + embed + upsert into pgvector."""
    from scripts.ingest import run_ingest
    n = run_ingest(data_dir=data_dir)
    typer.echo(f"upserted {n} chunks")


@app.command("build-kg")
def build_kg(limit: int = typer.Option(None)):
    """Extract entities/relations from chunks and load Neo4j."""
    from scripts.build_kg import run_build_kg
    e, r = run_build_kg(limit=limit)
    typer.echo(f"entities={e} relations={r}")


@app.command("build-noderag")
def build_noderag(
    limit: int = typer.Option(None, help="Cap number of chunks (for testing)"),
    skip_communities: bool = typer.Option(False, "--skip-communities"),
):
    """Build the heterogeneous-node NodeRAG index (semantic_units, attributes,
    communities). Requires build-kg to have run first."""
    from scripts.build_noderag import run_build_noderag
    out = run_build_noderag(limit=limit, skip_communities=skip_communities)
    typer.echo(out)


@app.command("noderag-stats")
def noderag_stats():
    """Print counts and a sample row per NodeRAG node type."""
    from rag.core.noderag_store import NODE_TYPES, NodeRAGStore
    nr = NodeRAGStore()
    total = nr.count()
    typer.echo(f"Total nodes: {total}")
    typer.echo("By type:")
    counts = nr.counts_by_type()
    for t in NODE_TYPES:
        typer.echo(f"  {t:<22} {counts.get(t, 0)}")
    typer.echo("\nSample (one per type):")
    for t in NODE_TYPES:
        with nr.conn.cursor() as cur:
            cur.execute(
                "SELECT id, content FROM noderag_nodes WHERE node_type = %s LIMIT 1;", (t,)
            )
            row = cur.fetchone()
        if row:
            nid, content = row
            preview = (content[:140] + "…") if len(content) > 140 else content
            typer.echo(f"\n  ▼ {t}  {nid}")
            typer.echo(f"    {preview}")
    nr.close()


@app.command()
def reset(confirm: bool = typer.Option(False, "--confirm")):
    """Wipe pgvector + Neo4j. Requires --confirm."""
    if not confirm:
        typer.echo("Refusing without --confirm.")
        raise typer.Exit(1)
    from rag.core.graph_store import GraphStore
    from rag.core.pg_store import PgStore
    pg = PgStore(); pg.init_schema(); pg.truncate(); pg.close()
    gs = GraphStore(); gs.init_schema(); gs.wipe(); gs.close()
    typer.echo("reset done")


if __name__ == "__main__":
    app()
