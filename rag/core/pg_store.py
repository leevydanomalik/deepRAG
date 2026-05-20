"""PostgreSQL + pgvector adapter for the rag_chunks table."""
from __future__ import annotations

import json
from typing import Any, Sequence

import psycopg
from pgvector.psycopg import register_vector

from rag.core.config import get_settings
from rag.core.loader import Chunk


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    chunk_index INT  NOT NULL,
    text        TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   VECTOR({dim}) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS rag_chunks_source_idx ON rag_chunks (source);
"""


class PgStore:
    def __init__(self, dsn: str | None = None):
        s = get_settings()
        self.dsn = dsn or s.pg_dsn
        self.dim = s.embedding_dim
        self._conn: psycopg.Connection | None = None

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, autocommit=True)
            register_vector(self._conn)
        return self._conn

    def init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_SQL.format(dim=self.dim))

    def truncate(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE rag_chunks;")

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> int:
        assert len(chunks) == len(embeddings), "chunks/embeddings length mismatch"
        rows = [
            (c.id, c.source, c.chunk_index, c.text, json.dumps(c.metadata), emb)
            for c, emb in zip(chunks, embeddings)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_chunks (id, source, chunk_index, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """,
                rows,
            )
        return len(rows)

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM rag_chunks;")
            return cur.fetchone()[0]

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source, chunk_index, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, query_embedding, k),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, chunk_index, text, metadata FROM rag_chunks WHERE id = ANY(%s);",
                (list(ids),),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
