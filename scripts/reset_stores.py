"""Wipe pgvector + Neo4j. Requires --confirm."""
from __future__ import annotations

import typer

from rag.core.graph_store import GraphStore
from rag.core.pg_store import PgStore

app = typer.Typer()


@app.command()
def main(confirm: bool = typer.Option(False, "--confirm")):
    if not confirm:
        typer.echo("Refusing to reset without --confirm.")
        raise typer.Exit(1)
    pg = PgStore()
    pg.init_schema()
    pg.truncate()
    pg.close()
    gs = GraphStore()
    gs.init_schema()
    gs.wipe()
    gs.close()
    typer.echo("Reset done.")


if __name__ == "__main__":
    app()
