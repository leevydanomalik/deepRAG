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
