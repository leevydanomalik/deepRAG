"""Walk data/ → chunk → embed → upsert into pgvector."""
from __future__ import annotations

from pathlib import Path

import typer

from rag.core.embeddings import embed_batch
from rag.core.loader import chunk_document, load_documents
from rag.core.pg_store import PgStore

app = typer.Typer()

DEFAULT_DATA_DIRS = ["data/seed", "data/raw"]
BATCH = 64


def run_ingest(data_dir: str | Path | None = None) -> int:
    store = PgStore()
    store.init_schema()

    roots: list[Path] = []
    if data_dir is not None:
        roots = [Path(data_dir)]
    else:
        roots = [Path(d) for d in DEFAULT_DATA_DIRS if Path(d).exists()]

    pending_chunks = []
    for root in roots:
        for source, text in load_documents(root):
            chunks = chunk_document(source=source, text=text)
            pending_chunks.extend(chunks)

    total = 0
    for i in range(0, len(pending_chunks), BATCH):
        batch = pending_chunks[i : i + BATCH]
        embeddings = embed_batch([c.text for c in batch])
        store.upsert(batch, embeddings)
        total += len(batch)
        typer.echo(f"  upserted {total}/{len(pending_chunks)}")

    typer.echo(f"Done. Total chunks: {total}")
    store.close()
    return total


@app.command()
def main(
    data_dir: str = typer.Option(None, help="Override default data/seed + data/raw"),
):
    """Ingest documents into pgvector."""
    run_ingest(data_dir=data_dir)


if __name__ == "__main__":
    app()
