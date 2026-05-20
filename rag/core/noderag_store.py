"""NodeRAG storage layer.

Adds a pgvector table `noderag_nodes` that holds heterogeneous nodes with
per-node embeddings. The graph topology (edges) is held in the existing
sqlite-graph DB used by Graph RAG, with `node_type` recorded in node
properties JSON. This lets us reuse the same edge store and run
Personalized PageRank over all node types.

Node types (Xu et al. 2025):
    entity                 — named things (Person, Org, Product, Concept, …)
    relationship           — relations promoted to first-class nodes
    semantic_unit          — atomic factual claims extracted from chunks
    attribute              — properties of entities
    high_level_element     — community-level concepts (from Louvain)
    high_level_overview    — LLM-generated summaries per community
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import psycopg
from pgvector.psycopg import register_vector

from rag.core.config import get_settings


NODE_TYPES = (
    "entity",
    "relationship",
    "semantic_unit",
    "attribute",
    "high_level_element",
    "high_level_overview",
)


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS noderag_nodes (
    id          TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   VECTOR({dim}) NOT NULL,
    chunk_ids   TEXT[] DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS noderag_nodes_embedding_idx
    ON noderag_nodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS noderag_nodes_type_idx ON noderag_nodes (node_type);
"""


class NodeRAGStore:
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
        sql = SCHEMA_SQL.replace("{dim}", str(self.dim))
        with self.conn.cursor() as cur:
            cur.execute(sql)

    def truncate(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE noderag_nodes;")

    def upsert_node(
        self,
        node_id: str,
        node_type: str,
        content: str,
        embedding: list[float],
        chunk_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"unknown node_type: {node_type!r}")
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO noderag_nodes (id, node_type, content, metadata, embedding, chunk_ids)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    chunk_ids = (
                        SELECT ARRAY(SELECT DISTINCT unnest(noderag_nodes.chunk_ids || EXCLUDED.chunk_ids))
                    );
                """,
                (
                    node_id,
                    node_type,
                    content,
                    json.dumps(metadata or {}),
                    embedding,
                    chunk_ids or [],
                ),
            )

    def count(self, node_type: str | None = None) -> int:
        with self.conn.cursor() as cur:
            if node_type is None:
                cur.execute("SELECT COUNT(*) FROM noderag_nodes;")
            else:
                cur.execute(
                    "SELECT COUNT(*) FROM noderag_nodes WHERE node_type = %s;", (node_type,)
                )
            return cur.fetchone()[0]

    def counts_by_type(self) -> dict[str, int]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT node_type, COUNT(*) FROM noderag_nodes GROUP BY node_type ORDER BY node_type;"
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    def vector_search(
        self,
        query_embedding: list[float],
        k: int = 12,
        node_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k cosine across (optionally filtered) node types."""
        if node_types:
            for t in node_types:
                if t not in NODE_TYPES:
                    raise ValueError(f"unknown node_type: {t!r}")
            sql = (
                "SELECT id, node_type, content, metadata, chunk_ids, "
                "1 - (embedding <=> %s::vector) AS score "
                "FROM noderag_nodes WHERE node_type = ANY(%s) "
                "ORDER BY embedding <=> %s::vector LIMIT %s;"
            )
            params = (query_embedding, node_types, query_embedding, k)
        else:
            sql = (
                "SELECT id, node_type, content, metadata, chunk_ids, "
                "1 - (embedding <=> %s::vector) AS score "
                "FROM noderag_nodes "
                "ORDER BY embedding <=> %s::vector LIMIT %s;"
            )
            params = (query_embedding, query_embedding, k)
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, node_type, content, metadata, chunk_ids FROM noderag_nodes WHERE id = ANY(%s);",
                (list(ids),),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def all_nodes_minimal(self) -> list[tuple[str, str]]:
        """(id, node_type) for every node — used to build the PPR graph."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, node_type FROM noderag_nodes;")
            return [(r[0], r[1]) for r in cur.fetchall()]

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
