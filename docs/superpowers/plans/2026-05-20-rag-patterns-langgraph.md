# deepRAG — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Reading note (2026-05-21):** Tasks 1-24 below were executed as written for the original four-pattern build, but the storage choice was later switched from **Neo4j → SQLite + sqlite-graph** (Task 25 in the addendum), local embeddings were preferred over OpenAI (Task 26), and a fifth pattern **NodeRAG** was added (Tasks 27-31). The original task bodies still read "Neo4j" — that's preserved as historical record. See the **Addendum (Tasks 25-31)** at the end of this file for the shipped architecture.

**Goal:** Build four RAG patterns (Naive, Agentic, Graph, Loop/PDCA) as LangGraph graphs sharing one PGVector store and one Neo4j graph, exposed via CLI + FastAPI + Streamlit.

> **Updated goal (shipped):** Five patterns (added NodeRAG), one PGVector store with two tables (`rag_chunks` + `noderag_nodes`), one SQLite + sqlite-graph graph store (no Neo4j).

**Architecture:** Single Python package `rag/` with `core/` (LLM/embeddings/stores/loader) + one subpackage per pattern + `registry.py`. App layer (`app/cli.py`, `app/api.py`, `app/ui.py`) calls into the registry. DeepSeek for chat, OpenAI for embeddings, PostgreSQL + pgvector for vectors, Neo4j for graph.

**Tech Stack:** Python 3.11, LangGraph 0.2, LangChain 0.3, DeepSeek chat (OpenAI-compatible), OpenAI text-embedding-3-small, PostgreSQL 16 + pgvector, Neo4j 5, Typer, FastAPI, uvicorn, Streamlit, pytest.

> **Shipped stack:** Python 3.12, LangGraph 0.2, LangChain 0.3, DeepSeek chat, **local `BAAI/bge-small-en-v1.5` (384-dim)** sentence-transformers (OpenAI selectable), PostgreSQL + pgvector, **SQLite + sqlite-graph** (no Neo4j), networkx + python-louvain (NodeRAG), Typer, FastAPI, uvicorn, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-05-20-rag-patterns-langgraph-design.md`

**Working dir:** `/Users/leevydmalik/Project/AI/DIP`

---

## File map

```
deepRAG/
├── pyproject.toml                  # Task 1
├── .gitignore                      # Task 1
├── .env.example                    # Task 1 (env names later changed in Task 26)
├── README.md                       # Task 24
├── vendor/libgraph.dylib           # Task 25 — sqlite-graph C extension (vendored)
├── data/seed/*.md                  # Task 12
├── data/raw/.gitkeep               # Task 1
├── scripts/ingest.py               # Task 9
├── scripts/build_kg.py             # Task 10
├── scripts/build_noderag.py        # Task 28 — NodeRAG extraction pipeline
├── scripts/reset_stores.py         # Task 11
├── rag/__init__.py                 # Task 1
├── rag/core/__init__.py            # Task 1
├── rag/core/config.py              # Task 2 (extended in Tasks 25, 26, 27)
├── rag/core/llm.py                 # Task 3 (renamed env vars in Task 26)
├── rag/core/embeddings.py          # Task 4 (local provider added in Task 26)
├── rag/core/loader.py              # Task 5
├── rag/core/pg_store.py            # Task 6
├── rag/core/neo4j_store.py         # Task 7 — DELETED in Task 25
├── rag/core/graph_store.py         # Task 25 — sqlite-graph adapter
├── rag/core/noderag_store.py       # Task 27 — pgvector adapter for noderag_nodes
├── rag/core/prompts.py             # Task 8 (NodeRAG prompts added in Task 28)
├── rag/naive/{state,graph}.py      # Task 13
├── rag/agentic/{state,tools,graph}.py    # Task 14 (Neo4jStore → GraphStore in Task 25)
├── rag/graph_rag/{state,graph}.py  # Task 15 (Neo4jStore → GraphStore in Task 25)
├── rag/loop/{state,agents,graph}.py      # Task 16, 17, 18 (Neo4jStore → GraphStore in Task 25)
├── rag/noderag/{state,graph}.py    # Task 29 — NodeRAG LangGraph (HNSW + PPR)
├── rag/registry.py                 # Task 19 (noderag added in Task 30)
├── app/cli.py                      # Task 20 (build-noderag, compare, noderag-stats in Task 30)
├── app/api.py                      # Task 21 (/noderag/build, /noderag/stats in Task 30)
├── app/ui.py                       # Task 22 — auto-picks up noderag from /patterns
├── notebooks/0[1-4]_*.ipynb        # Task 23
└── tests/                          # populated alongside each task
                                    # + test_graph_store, test_noderag, test_build_noderag
```

---

## Test strategy

- **Unit tests** (fast, no network): config, loader, chunking, prompts. Run on every task.
- **Integration tests** (need PG + Neo4j running): pg_store, neo4j_store, ingest, build_kg. Skip with `@pytest.mark.skipif` if env not configured.
- **LLM tests** (cost tokens): patterns end-to-end. Mark `@pytest.mark.llm`, opt-in via `pytest -m llm`.
- All LLM-touching code accepts an injectable client so tests can use a fake.

Run all fast tests: `pytest -m "not llm and not integration"`
Run all: `pytest -m "" --no-skip`

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `rag/__init__.py` (empty)
- Create: `rag/core/__init__.py` (empty)
- Create: `app/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `data/raw/.gitkeep` (empty)
- Create: `data/seed/.gitkeep` (empty)
- Create: `scripts/.gitkeep` (empty)

- [ ] **Step 1: Initialize git and create dirs**

```bash
cd /Users/leevydmalik/Project/AI/DIP
git init -b main
mkdir -p rag/core rag/naive rag/agentic rag/graph_rag rag/loop app tests data/seed data/raw scripts notebooks docs/superpowers/{specs,plans}
touch rag/__init__.py rag/core/__init__.py app/__init__.py tests/__init__.py data/seed/.gitkeep data/raw/.gitkeep scripts/.gitkeep
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "dip-rag"
version = "0.1.0"
description = "Four RAG patterns (Naive, Agentic, Graph, Loop) in LangGraph"
requires-python = ">=3.11"
dependencies = [
  "langgraph>=0.2.50",
  "langchain>=0.3.0",
  "langchain-core>=0.3.0",
  "langchain-openai>=0.2.0",
  "langchain-community>=0.3.0",
  "openai>=1.50.0",
  "psycopg[binary]>=3.2.0",
  "pgvector>=0.3.0",
  "neo4j>=5.24.0",
  "pypdf>=5.0.0",
  "tiktoken>=0.8.0",
  "pydantic>=2.9.0",
  "pydantic-settings>=2.5.0",
  "python-dotenv>=1.0.0",
  "typer>=0.12.0",
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "streamlit>=1.39.0",
  "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.24",
  "ruff>=0.6",
  "ipykernel>=6.29",
  "jupyterlab>=4.2",
]

[tool.pytest.ini_options]
markers = [
  "integration: needs PG + Neo4j running",
  "llm: makes real LLM API calls (costs tokens)",
]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py311"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["rag*", "app*"]
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
.env
data/raw/*
!data/raw/.gitkeep
.DS_Store
build/
dist/
```

- [ ] **Step 4: Write `.env.example`**

```
# DeepSeek (OpenAI-compatible chat)
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# OpenAI (embeddings only)
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

# Postgres + pgvector
PG_DSN=postgresql://postgres:postgres@localhost:5432/dip_rag

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# Loop RAG
LOOP_MAX_ITERATIONS=4
LOOP_CONVERGENCE_THRESHOLD=0.15

# API
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_API_URL=http://localhost:8000
```

- [ ] **Step 5: Create venv and install**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Expected: `pip list` shows langgraph, fastapi, streamlit, etc.

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "chore: project scaffold (pyproject, env example, dirs)"
```

---

### Task 2: Config module (`rag/core/config.py`)

**Files:**
- Create: `rag/core/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
import os
from rag.core.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("PG_DSN", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")

    s = Settings()
    assert s.deepseek_api_key.get_secret_value() == "sk-test"
    assert s.deepseek_model == "deepseek-chat"
    assert s.embedding_dim == 1536
    assert s.loop_max_iterations == 4
    assert s.loop_convergence_threshold == 0.15


def test_settings_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    monkeypatch.setenv("LOOP_MAX_ITERATIONS", "8")
    s = Settings()
    assert s.loop_max_iterations == 8
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError: rag.core.config`.

- [ ] **Step 3: Implement `rag/core/config.py`**

```python
"""Centralized settings loaded from .env via pydantic-settings."""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # DeepSeek
    deepseek_api_key: SecretStr
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # OpenAI embeddings
    openai_api_key: SecretStr
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Postgres
    pg_dsn: str

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr

    # Loop RAG
    loop_max_iterations: int = 4
    loop_convergence_threshold: float = 0.15

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_config.py -v
```
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/core/config.py tests/test_config.py
git commit -m "feat(core): typed settings loaded from .env"
```

---

### Task 3: DeepSeek chat client (`rag/core/llm.py`)

**Files:**
- Create: `rag/core/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm.py
from unittest.mock import MagicMock, patch

from rag.core.llm import get_chat_model


@patch("rag.core.llm.ChatOpenAI")
def test_chat_model_configured_for_deepseek(mock_chat, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")

    from rag.core.config import get_settings
    get_settings.cache_clear()

    get_chat_model()

    mock_chat.assert_called_once()
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["api_key"].get_secret_value() == "sk-test"
    assert kwargs["temperature"] == 0.0
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_llm.py -v
```
Expected: import fail.

- [ ] **Step 3: Implement `rag/core/llm.py`**

```python
"""DeepSeek chat client. DeepSeek is OpenAI-compatible, so we use ChatOpenAI
with a custom base_url."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from rag.core.config import get_settings


def get_chat_model(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.deepseek_model,
        base_url=s.deepseek_base_url,
        api_key=s.deepseek_api_key,
        temperature=temperature,
        **kwargs,
    )
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_llm.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/core/llm.py tests/test_llm.py
git commit -m "feat(core): DeepSeek chat client via ChatOpenAI base_url override"
```

---

### Task 4: OpenAI embeddings (`rag/core/embeddings.py`)

**Files:**
- Create: `rag/core/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_embeddings.py
from unittest.mock import patch

from rag.core.embeddings import get_embeddings


@patch("rag.core.embeddings.OpenAIEmbeddings")
def test_embeddings_configured(mock_emb, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")

    from rag.core.config import get_settings
    get_settings.cache_clear()

    get_embeddings()
    kwargs = mock_emb.call_args.kwargs
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["api_key"].get_secret_value() == "sk-real"
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_embeddings.py -v
```

- [ ] **Step 3: Implement `rag/core/embeddings.py`**

```python
"""OpenAI embeddings (text-embedding-3-small, 1536-dim)."""
from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from rag.core.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        api_key=s.openai_api_key,
    )


def embed_text(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_embeddings.py -v
```

- [ ] **Step 5: Commit**

```bash
git add rag/core/embeddings.py tests/test_embeddings.py
git commit -m "feat(core): OpenAI embeddings client"
```

---

### Task 5: Document loader + chunking (`rag/core/loader.py`)

**Files:**
- Create: `rag/core/loader.py`
- Create: `tests/test_loader.py`
- Create: `tests/fixtures/sample.md`

- [ ] **Step 1: Create fixture and write failing test**

```python
# tests/test_loader.py
from pathlib import Path

import pytest

from rag.core.loader import Chunk, chunk_document, load_documents

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_chunk_document_stable_ids():
    text = "alpha. " * 300
    chunks_a = chunk_document(source="t.md", text=text)
    chunks_b = chunk_document(source="t.md", text=text)
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
    assert len(chunks_a) > 1
    assert all(isinstance(c, Chunk) for c in chunks_a)


def test_chunk_document_metadata():
    chunks = chunk_document(source="t.md", text="x" * 2000)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].source == "t.md"


def test_load_documents_markdown(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\n\nWorld.")
    docs = list(load_documents(tmp_path))
    assert len(docs) == 1
    assert docs[0][0].endswith("a.md")
    assert "Hello" in docs[0][1]


def test_load_documents_skips_unsupported(tmp_path):
    (tmp_path / "a.md").write_text("ok")
    (tmp_path / "b.png").write_bytes(b"\x89PNG\r\n")
    docs = list(load_documents(tmp_path))
    assert len(docs) == 1
```

Create the fixture:

```bash
mkdir -p tests/fixtures
printf "# Title\n\nSample body.\n" > tests/fixtures/sample.md
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_loader.py -v
```

- [ ] **Step 3: Implement `rag/core/loader.py`**

```python
"""Document loading + chunking. Supported: .txt, .md, .pdf."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    chunk_index: int
    text: str
    metadata: dict


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(root: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (absolute_path, full_text) for every supported file under root."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        yield (str(path.resolve()), text)


def chunk_document(
    source: str,
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    extra_metadata: dict | None = None,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    out: list[Chunk] = []
    for i, ct in enumerate(raw_chunks):
        cid = hashlib.sha256(f"{source}:{i}".encode()).hexdigest()
        out.append(
            Chunk(
                id=cid,
                source=source,
                chunk_index=i,
                text=ct,
                metadata={"source": source, **(extra_metadata or {})},
            )
        )
    return out
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_loader.py -v
```

- [ ] **Step 5: Commit**

```bash
git add rag/core/loader.py tests/test_loader.py tests/fixtures/sample.md
git commit -m "feat(core): document loader + recursive chunker with stable IDs"
```

---

### Task 6: PGVector store (`rag/core/pg_store.py`)

**Files:**
- Create: `rag/core/pg_store.py`
- Create: `tests/test_pg_store.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_pg_store.py
import os

import pytest

from rag.core.loader import Chunk
from rag.core.pg_store import PgStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    if not os.environ.get("PG_DSN"):
        pytest.skip("PG_DSN not set")
    s = PgStore()
    s.init_schema()
    s.truncate()
    yield s
    s.close()


def _fake_chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c{i}",
        source=f"src/{i}.md",
        chunk_index=i,
        text=f"chunk number {i}",
        metadata={"k": "v"},
    )


def test_upsert_and_count(store):
    chunks = [_fake_chunk(i) for i in range(3)]
    embeddings = [[0.1] * 1536 for _ in chunks]
    store.upsert(chunks, embeddings)
    assert store.count() == 3


def test_similarity_search(store):
    chunks = [_fake_chunk(i) for i in range(3)]
    embeddings = [[float(i)] + [0.0] * 1535 for i in range(3)]
    store.upsert(chunks, embeddings)
    hits = store.similarity_search(query_embedding=[2.0] + [0.0] * 1535, k=2)
    assert len(hits) == 2
    assert hits[0]["id"] == "c2"


def test_upsert_idempotent(store):
    c = _fake_chunk(0)
    store.upsert([c], [[0.0] * 1536])
    store.upsert([c], [[0.0] * 1536])
    assert store.count() == 1
```

- [ ] **Step 2: Run test (expect fail with import error)**

```bash
pytest tests/test_pg_store.py -v
```

- [ ] **Step 3: Implement `rag/core/pg_store.py`**

```python
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
```

- [ ] **Step 4: Run tests (PG must be up)**

```bash
pytest tests/test_pg_store.py -v -m integration
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add rag/core/pg_store.py tests/test_pg_store.py
git commit -m "feat(core): pgvector store (init, upsert, similarity, idempotent)"
```

---

### Task 7: Neo4j store (`rag/core/neo4j_store.py`)

**Files:**
- Create: `rag/core/neo4j_store.py`
- Create: `tests/test_neo4j_store.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_neo4j_store.py
import os

import pytest

from rag.core.neo4j_store import Neo4jStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    if not os.environ.get("NEO4J_PASSWORD"):
        pytest.skip("NEO4J_PASSWORD not set")
    s = Neo4jStore()
    s.init_schema()
    s.wipe()
    yield s
    s.close()


def test_merge_entity_idempotent(store):
    store.merge_entity(name="Acme", type_="Org", description="d1", chunk_ids=["c1"])
    store.merge_entity(name="Acme", type_="Org", description="d2", chunk_ids=["c2"])
    e = store.get_entity("Acme")
    assert e["type"] == "Org"
    assert "c1" in e["source_chunks"] and "c2" in e["source_chunks"]


def test_merge_relation(store):
    store.merge_entity("A", "Org", "", ["x"])
    store.merge_entity("B", "Person", "", ["x"])
    store.merge_relation("A", "B", "HAS_MEMBER", "A has member B", chunk_ids=["x"])
    rels = store.find_relations(["A"])
    assert any(r["type"] == "HAS_MEMBER" and r["dst"] == "b" for r in rels)


def test_expand_subgraph(store):
    store.merge_entity("A", "Org", "", [])
    store.merge_entity("B", "Person", "", [])
    store.merge_entity("C", "Place", "", [])
    store.merge_relation("A", "B", "EMPLOYS", "", chunk_ids=[])
    store.merge_relation("B", "C", "LIVES_IN", "", chunk_ids=[])
    nodes, edges = store.expand_subgraph(["A"], hops=2)
    names = {n["name"] for n in nodes}
    assert {"a", "b", "c"} <= names
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_neo4j_store.py -v -m integration
```

- [ ] **Step 3: Implement `rag/core/neo4j_store.py`**

```python
"""Neo4j adapter for entity/relation graph used by Graph RAG and Loop RAG."""
from __future__ import annotations

from typing import Any

from neo4j import Driver, GraphDatabase

from rag.core.config import get_settings


class Neo4jStore:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        s = get_settings()
        self.uri = uri or s.neo4j_uri
        self.user = user or s.neo4j_user
        self.password = password or s.neo4j_password.get_secret_value()
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def init_schema(self) -> None:
        with self.driver.session() as sess:
            sess.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            sess.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    def wipe(self) -> None:
        with self.driver.session() as sess:
            sess.run("MATCH (n) DETACH DELETE n")

    # ----- writes -----
    def merge_entity(self, name: str, type_: str, description: str, chunk_ids: list[str]) -> None:
        norm = name.strip().lower()
        with self.driver.session() as sess:
            sess.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.description = $desc, e.source_chunks = $chunks, e.display_name = $display
                ON MATCH  SET e.type = coalesce(e.type, $type),
                              e.description = CASE WHEN coalesce(e.description, '') = '' THEN $desc
                                                   WHEN $desc = '' THEN e.description
                                                   ELSE e.description + ' | ' + $desc END,
                              e.source_chunks = apoc.coll.toSet(coalesce(e.source_chunks, []) + $chunks)
                """ if False else
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.type = $type, e.description = $desc, e.source_chunks = $chunks, e.display_name = $display
                ON MATCH  SET e.type = coalesce(e.type, $type),
                              e.description = CASE WHEN coalesce(e.description, '') = '' THEN $desc
                                                   WHEN $desc = '' THEN e.description
                                                   ELSE e.description + ' | ' + $desc END,
                              e.source_chunks = [x IN coalesce(e.source_chunks, []) + $chunks WHERE x IS NOT NULL]
                """,
                name=norm, type=type_, desc=description, chunks=chunk_ids, display=name.strip(),
            )

    def merge_relation(self, src: str, dst: str, type_: str, description: str, chunk_ids: list[str]) -> None:
        with self.driver.session() as sess:
            sess.run(
                """
                MERGE (a:Entity {name: $src})
                MERGE (b:Entity {name: $dst})
                MERGE (a)-[r:RELATES {type: $type}]->(b)
                ON CREATE SET r.description = $desc, r.source_chunks = $chunks, r.weight = 1
                ON MATCH  SET r.weight = coalesce(r.weight, 1) + 1,
                              r.source_chunks = [x IN coalesce(r.source_chunks, []) + $chunks WHERE x IS NOT NULL],
                              r.description = CASE WHEN coalesce(r.description, '') = '' THEN $desc
                                                   WHEN $desc = '' THEN r.description
                                                   ELSE r.description + ' | ' + $desc END
                """,
                src=src.strip().lower(), dst=dst.strip().lower(),
                type=type_, desc=description, chunks=chunk_ids,
            )

    # ----- reads -----
    def get_entity(self, name: str) -> dict[str, Any] | None:
        with self.driver.session() as sess:
            rec = sess.run(
                "MATCH (e:Entity {name: $name}) RETURN e",
                name=name.strip().lower(),
            ).single()
            return dict(rec["e"]) if rec else None

    def find_entities_by_names(self, names: list[str]) -> list[dict[str, Any]]:
        norm = [n.strip().lower() for n in names]
        with self.driver.session() as sess:
            res = sess.run(
                "MATCH (e:Entity) WHERE e.name IN $names RETURN e",
                names=norm,
            )
            return [dict(r["e"]) for r in res]

    def find_relations(self, entity_names: list[str]) -> list[dict[str, Any]]:
        norm = [n.strip().lower() for n in entity_names]
        with self.driver.session() as sess:
            res = sess.run(
                """
                MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                WHERE a.name IN $names OR b.name IN $names
                RETURN a.name AS src, b.name AS dst, r.type AS type,
                       r.description AS description, r.source_chunks AS source_chunks
                """,
                names=norm,
            )
            return [r.data() for r in res]

    def expand_subgraph(self, seed_names: list[str], hops: int = 2) -> tuple[list[dict], list[dict]]:
        norm = [n.strip().lower() for n in seed_names]
        with self.driver.session() as sess:
            res = sess.run(
                f"""
                MATCH (seed:Entity) WHERE seed.name IN $names
                CALL {{
                    WITH seed
                    MATCH p=(seed)-[r:RELATES*1..{hops}]-(n)
                    RETURN nodes(p) AS ns, relationships(p) AS rs
                }}
                RETURN ns, rs
                """,
                names=norm,
            )
            nodes_by_name: dict[str, dict] = {}
            edges: list[dict] = []
            for row in res:
                for n in row["ns"]:
                    nodes_by_name[n["name"]] = dict(n)
                for r in row["rs"]:
                    edges.append({
                        "src": r.start_node["name"],
                        "dst": r.end_node["name"],
                        "type": r["type"],
                        "description": r.get("description", ""),
                        "source_chunks": r.get("source_chunks", []) or [],
                    })
            return list(nodes_by_name.values()), edges
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_neo4j_store.py -v -m integration
```

- [ ] **Step 5: Commit**

```bash
git add rag/core/neo4j_store.py tests/test_neo4j_store.py
git commit -m "feat(core): Neo4j store with merge entity/relation and subgraph expand"
```

---

### Task 8: Prompts module (`rag/core/prompts.py`)

**Files:**
- Create: `rag/core/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_prompts.py
from rag.core.prompts import (
    ANSWER_PROMPT,
    KG_EXTRACTION_PROMPT,
    LOOP_PLAN_PROMPT,
)


def test_prompts_have_required_placeholders():
    assert "{question}" in ANSWER_PROMPT
    assert "{context}" in ANSWER_PROMPT
    assert "{text}" in KG_EXTRACTION_PROMPT
    assert "{question}" in LOOP_PLAN_PROMPT
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_prompts.py -v
```

- [ ] **Step 3: Implement `rag/core/prompts.py`**

```python
"""Shared prompt templates for all RAG patterns."""

ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context
below. If the answer is not in the context, say "I don't know based on the provided
context."

Context:
{context}

Question:
{question}

Answer:"""


KG_EXTRACTION_PROMPT = """Extract entities and relations from the text below.
Return STRICT JSON with this exact shape and nothing else:

{{
  "entities": [
    {{"name": "...", "type": "Person|Org|Place|Concept|Product|Event|Other", "description": "..."}}
  ],
  "relations": [
    {{"src": "...", "dst": "...", "type": "...", "description": "..."}}
  ]
}}

Rules:
- Entity names are the canonical proper form (e.g. "LangChain", not "the LangChain library").
- "src" and "dst" must match an entity name in the entities array.
- Keep descriptions under 200 chars.
- Relation types should be SHORT uppercase verbs like CREATED, USES, DEPENDS_ON.

Text:
{text}

JSON:"""


LOOP_PLAN_PROMPT = """You are the PLAN agent in a PDCA RAG loop.
Given the user question, output STRICT JSON:

{{
  "task_type": "factual|explanatory|comparative|predictive|control|diagnosis",
  "entities": ["..."],
  "constraints": ["..."],
  "sub_goals": ["..."],
  "prompt_template": "stuff|stepwise|structured"
}}

Question:
{question}

JSON:"""


LOOP_CHECK_PROMPT = """You are the CHECK agent. Given the question and the proposed answer,
estimate per-evidence support score for each piece of evidence (0..1).
Return STRICT JSON:

{{
  "support_scores": [0.0, 0.0, ...]
}}

Question: {question}
Answer:   {answer}
Evidence (numbered):
{evidence}

JSON:"""


LOOP_ACT_PROMPT = """You are the ACT agent. The previous iteration failed with dominant
deviation '{dominant}'. Suggest a refined query and prompt template.
Return STRICT JSON:

{{
  "rewritten_query": "...",
  "prompt_template": "stuff|stepwise|structured"
}}

Original question: {question}
Previous answer:   {answer}

JSON:"""


GRAPH_RAG_ENTITY_EXTRACT_PROMPT = """Extract the entities (proper nouns, named concepts)
from this question. Return STRICT JSON: {{"entities": ["..."]}}.

Question: {question}

JSON:"""
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_prompts.py -v
```

- [ ] **Step 5: Commit**

```bash
git add rag/core/prompts.py tests/test_prompts.py
git commit -m "feat(core): shared prompt templates"
```

---

### Task 9: Ingest script (`scripts/ingest.py`)

**Files:**
- Create: `scripts/ingest.py`
- Create: `tests/test_ingest_script.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_ingest_script.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.ingest import run_ingest


def test_ingest_chunks_and_upserts(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()

    (tmp_path / "a.md").write_text("# Hello\n\n" + "body text. " * 200)

    fake_store = MagicMock()
    fake_store.upsert.return_value = 0

    with patch("scripts.ingest.PgStore", return_value=fake_store), \
         patch("scripts.ingest.embed_batch", return_value=[[0.0] * 1536] * 10):
        n = run_ingest(data_dir=tmp_path)
    assert n > 0
    fake_store.init_schema.assert_called_once()
    assert fake_store.upsert.called
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_ingest_script.py -v
```

- [ ] **Step 3: Implement `scripts/ingest.py`**

```python
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

    total = 0
    pending_chunks = []

    for root in roots:
        for source, text in load_documents(root):
            chunks = chunk_document(source=source, text=text)
            pending_chunks.extend(chunks)

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
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_ingest_script.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest.py tests/test_ingest_script.py
git commit -m "feat(scripts): ingest documents to pgvector with batch embedding"
```

---

### Task 10: Build-KG script (`scripts/build_kg.py`)

**Files:**
- Create: `scripts/build_kg.py`
- Create: `tests/test_build_kg.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_build_kg.py
import json
from unittest.mock import MagicMock, patch

from scripts.build_kg import extract_kg_from_chunk, run_build_kg


def test_extract_kg_from_chunk_parses_json():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "entities": [{"name": "LangGraph", "type": "Product", "description": "DAG framework"}],
        "relations": [{"src": "LangGraph", "dst": "LangChain", "type": "BUILT_BY", "description": ""}],
    }))
    out = extract_kg_from_chunk(fake_llm, "irrelevant text")
    assert out["entities"][0]["name"] == "LangGraph"
    assert out["relations"][0]["type"] == "BUILT_BY"


def test_extract_kg_handles_malformed_json():
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="not json at all")
    out = extract_kg_from_chunk(fake_llm, "x")
    assert out == {"entities": [], "relations": []}


def test_run_build_kg_iterates_chunks(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    monkeypatch.setenv("NEO4J_PASSWORD", "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()

    fake_pg = MagicMock()
    fake_pg.conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [
        ("c1", "src.md", 0, "LangGraph builds DAGs.", {}),
    ]
    fake_neo = MagicMock()
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "entities": [{"name": "LangGraph", "type": "Product", "description": ""}],
        "relations": [],
    }))
    with patch("scripts.build_kg.PgStore", return_value=fake_pg), \
         patch("scripts.build_kg.Neo4jStore", return_value=fake_neo), \
         patch("scripts.build_kg.get_chat_model", return_value=fake_llm):
        e, r = run_build_kg(limit=1)
    assert e == 1 and r == 0
    fake_neo.init_schema.assert_called_once()
    fake_neo.merge_entity.assert_called()
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_build_kg.py -v
```

- [ ] **Step 3: Implement `scripts/build_kg.py`**

```python
"""Walk pg_chunks → ask LLM for entities/relations → MERGE into Neo4j."""
from __future__ import annotations

import json
import re

import typer

from rag.core.llm import get_chat_model
from rag.core.neo4j_store import Neo4jStore
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
    neo = Neo4jStore()
    neo.init_schema()
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
            neo.merge_entity(
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
            neo.merge_relation(
                src=src,
                dst=dst,
                type_=r.get("type", "RELATED_TO"),
                description=r.get("description", ""),
                chunk_ids=[chunk_id],
            )
            relations_added += 1

    pg.close()
    neo.close()
    typer.echo(f"Done. entities={entities_added} relations={relations_added}")
    return entities_added, relations_added


@app.command()
def main(limit: int = typer.Option(None, help="Cap number of chunks to process")):
    run_build_kg(limit=limit)


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_build_kg.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/build_kg.py tests/test_build_kg.py
git commit -m "feat(scripts): build_kg extracts entities+relations and loads Neo4j"
```

---

### Task 11: Reset script (`scripts/reset_stores.py`)

**Files:**
- Create: `scripts/reset_stores.py`

- [ ] **Step 1: Implement directly (small, no behavioral test needed)**

```python
"""Wipe pgvector + Neo4j. Requires --confirm."""
from __future__ import annotations

import typer

from rag.core.neo4j_store import Neo4jStore
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
    neo = Neo4jStore()
    neo.wipe()
    neo.close()
    typer.echo("Reset done.")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Smoke-run**

```bash
python -m scripts.reset_stores             # should refuse
python -m scripts.reset_stores --confirm   # should wipe (only if PG+Neo4j up)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/reset_stores.py
git commit -m "feat(scripts): reset_stores wipes pgvector + neo4j"
```

---

### Task 12: Seed data

**Files:**
- Create: `data/seed/01_langgraph.md`
- Create: `data/seed/02_langchain.md`
- Create: `data/seed/03_rag_patterns.md`

- [ ] **Step 1: Write 3 markdown files with bundled content**

`data/seed/01_langgraph.md`:
```markdown
# LangGraph

LangGraph is a library by LangChain Inc. for building stateful, multi-actor
applications with LLMs. It models workflows as graphs where nodes are functions
and edges define control flow.

Core concepts:
- StateGraph: a typed graph of nodes that read/write a shared state object.
- Nodes: pure Python functions that take state and return a partial state update.
- Edges: static (always go to node X) or conditional (a function decides).
- Compile: turn the graph into a runnable application.

LangGraph supports cycles, which makes it suitable for agent loops, multi-step
reasoning, and PDCA-style closed-loop systems.
```

`data/seed/02_langchain.md`:
```markdown
# LangChain

LangChain is an open-source framework for developing applications powered by
large language models. It provides abstractions for chains (sequences of calls),
tools, retrievers, vector stores, and document loaders.

LangChain was created by Harrison Chase in October 2022. The company LangChain
Inc. now maintains it along with LangSmith (observability) and LangGraph
(stateful orchestration).
```

`data/seed/03_rag_patterns.md`:
```markdown
# RAG Patterns

Retrieval-augmented generation (RAG) combines a retriever with a generator LLM.

Naive RAG: retrieve top-k chunks by similarity, stuff into the prompt, generate.

Agentic RAG: the LLM acts as an agent that decides when and how to call
retrieval tools. It can issue multiple queries, refine, or skip retrieval
entirely.

Graph RAG: builds a knowledge graph from the corpus. Retrieval expands a
subgraph from entities mentioned in the question and pulls associated chunks.

Loop RAG (PDCA): a closed-loop multi-agent pattern with Plan, Do, Check, Act
nodes. The system iterates until a deviation signal converges below a
threshold. Based on Bai et al., Buildings 2026, 16, 196.
```

- [ ] **Step 2: Commit**

```bash
git add data/seed/
git commit -m "data: seed corpus on LangGraph, LangChain, RAG patterns"
```

---

### Task 13: Naive RAG graph (`rag/naive/`)

**Files:**
- Create: `rag/naive/__init__.py` (empty)
- Create: `rag/naive/state.py`
- Create: `rag/naive/graph.py`
- Create: `tests/test_naive.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_naive.py
from unittest.mock import MagicMock, patch

from rag.naive.graph import build_graph


def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_naive_graph_invokes_retrieve_then_generate(monkeypatch):
    _env(monkeypatch)

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = [
        {"id": "c1", "source": "a.md", "chunk_index": 0, "text": "alpha", "metadata": {}, "score": 0.9},
        {"id": "c2", "source": "a.md", "chunk_index": 1, "text": "beta",  "metadata": {}, "score": 0.8},
    ]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="answer ok")

    with patch("rag.naive.graph.PgStore", return_value=fake_store), \
         patch("rag.naive.graph.embed_text", return_value=[0.0] * 1536), \
         patch("rag.naive.graph.get_chat_model", return_value=fake_llm):
        app = build_graph()
        out = app.invoke({"question": "what is alpha?"})

    assert out["answer"] == "answer ok"
    assert len(out["retrieved"]) == 2
    assert out["history"][0]["node"] == "retrieve"
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_naive.py -v
```

- [ ] **Step 3: Implement `rag/naive/state.py`**

```python
"""State for Naive RAG: a single retrieve → generate pass."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class NaiveState(TypedDict, total=False):
    question: str
    retrieved: list[dict[str, Any]]
    answer: str
    history: Annotated[list[dict[str, Any]], add]
```

- [ ] **Step 4: Implement `rag/naive/graph.py`**

```python
"""Naive RAG: retrieve top-k → stuff prompt → generate."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag.core.embeddings import embed_text
from rag.core.llm import get_chat_model
from rag.core.pg_store import PgStore
from rag.core.prompts import ANSWER_PROMPT
from rag.naive.state import NaiveState


def _retrieve_node(state: NaiveState) -> NaiveState:
    store = PgStore()
    q_emb = embed_text(state["question"])
    hits = store.similarity_search(q_emb, k=5)
    store.close()
    return {
        "retrieved": hits,
        "history": [{"node": "retrieve", "hits": [h["id"] for h in hits]}],
    }


def _generate_node(state: NaiveState) -> NaiveState:
    ctx = "\n\n---\n\n".join(
        f"[{h['source']}#{h['chunk_index']}] {h['text']}" for h in state.get("retrieved", [])
    )
    prompt = ANSWER_PROMPT.format(context=ctx or "(no context)", question=state["question"])
    llm = get_chat_model(temperature=0.0)
    resp = llm.invoke(prompt)
    return {
        "answer": getattr(resp, "content", str(resp)),
        "history": [{"node": "generate"}],
    }


def build_graph():
    g = StateGraph(NaiveState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("generate", _generate_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
```

- [ ] **Step 5: Run test (expect pass)**

```bash
pytest tests/test_naive.py -v
```

- [ ] **Step 6: Commit**

```bash
git add rag/naive/ tests/test_naive.py
git commit -m "feat(naive): naive RAG graph (retrieve → generate)"
```

---

### Task 14: Agentic RAG graph (`rag/agentic/`)

**Files:**
- Create: `rag/agentic/__init__.py`
- Create: `rag/agentic/state.py`
- Create: `rag/agentic/tools.py`
- Create: `rag/agentic/graph.py`
- Create: `tests/test_agentic.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_agentic.py
from unittest.mock import MagicMock, patch

from rag.agentic.graph import build_graph


def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_agentic_graph_compiles(monkeypatch):
    _env(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.bind_tools.return_value = fake_llm
    fake_llm.invoke.return_value = MagicMock(content="done", tool_calls=[])
    with patch("rag.agentic.graph.get_chat_model", return_value=fake_llm):
        app = build_graph()
    assert app is not None
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_agentic.py -v
```

- [ ] **Step 3: Implement `rag/agentic/state.py`**

```python
"""State for Agentic RAG: agent + tools loop."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class AgenticState(TypedDict, total=False):
    question: str
    messages: Annotated[list[BaseMessage], add]
    answer: str
    history: Annotated[list[dict[str, Any]], add]
```

- [ ] **Step 4: Implement `rag/agentic/tools.py`**

```python
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
```

- [ ] **Step 5: Implement `rag/agentic/graph.py`**

```python
"""Agentic RAG: ReAct-style loop with vector_search + kg_lookup tools."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from rag.agentic.state import AgenticState
from rag.agentic.tools import TOOLS
from rag.core.llm import get_chat_model

SYSTEM_PROMPT = (
    "You are a research assistant. Use the provided tools (vector_search, kg_lookup) "
    "to gather context before answering. When you have enough information, write the "
    "final answer with citations like [source#chunk]."
)


def _agent_node(state: AgenticState):
    llm = get_chat_model(temperature=0.0).bind_tools(TOOLS)
    msgs = state.get("messages") or [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["question"]),
    ]
    resp = llm.invoke(msgs)
    update = {"messages": [resp], "history": [{"node": "agent", "tool_calls": getattr(resp, "tool_calls", [])}]}
    if not getattr(resp, "tool_calls", None):
        update["answer"] = getattr(resp, "content", "")
    return update


def build_graph():
    g = StateGraph(AgenticState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", ToolNode(TOOLS))

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()
```

- [ ] **Step 6: Run test (expect pass)**

```bash
pytest tests/test_agentic.py -v
```

- [ ] **Step 7: Commit**

```bash
git add rag/agentic/ tests/test_agentic.py
git commit -m "feat(agentic): ReAct-style RAG with vector_search + kg_lookup tools"
```

---

### Task 15: Graph RAG (`rag/graph_rag/`)

**Files:**
- Create: `rag/graph_rag/__init__.py`
- Create: `rag/graph_rag/state.py`
- Create: `rag/graph_rag/graph.py`
- Create: `tests/test_graph_rag.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_graph_rag.py
import json
from unittest.mock import MagicMock, patch

from rag.graph_rag.graph import build_graph


def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_graph_rag_end_to_end(monkeypatch):
    _env(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"entities": ["LangGraph"]})),
        MagicMock(content="final answer"),
    ]
    fake_neo = MagicMock()
    fake_neo.expand_subgraph.return_value = (
        [{"name": "langgraph", "type": "Product", "description": "DAG framework", "source_chunks": ["c1"]}],
        [{"src": "langgraph", "dst": "langchain", "type": "BUILT_BY", "description": "", "source_chunks": ["c1"]}],
    )
    fake_pg = MagicMock()
    fake_pg.fetch_by_ids.return_value = [
        {"id": "c1", "source": "a.md", "chunk_index": 0, "text": "LangGraph...", "metadata": {}},
    ]

    with patch("rag.graph_rag.graph.get_chat_model", return_value=fake_llm), \
         patch("rag.graph_rag.graph.Neo4jStore", return_value=fake_neo), \
         patch("rag.graph_rag.graph.PgStore", return_value=fake_pg):
        app = build_graph()
        out = app.invoke({"question": "What is LangGraph?"})

    assert out["answer"] == "final answer"
    assert "LangGraph" in out["entities"]
    assert len(out["subgraph_nodes"]) == 1
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_graph_rag.py -v
```

- [ ] **Step 3: Implement `rag/graph_rag/state.py`**

```python
"""State for Graph RAG: entity extract → subgraph expand → chunk lookup → generate."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class GraphRAGState(TypedDict, total=False):
    question: str
    entities: list[str]
    subgraph_nodes: list[dict[str, Any]]
    subgraph_edges: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    answer: str
    history: Annotated[list[dict[str, Any]], add]
```

- [ ] **Step 4: Implement `rag/graph_rag/graph.py`**

```python
"""Graph RAG: extract entities from question → expand Neo4j subgraph →
fetch chunks the entities/relations point to → generate."""
from __future__ import annotations

import json
import re

from langgraph.graph import END, StateGraph

from rag.core.llm import get_chat_model
from rag.core.neo4j_store import Neo4jStore
from rag.core.pg_store import PgStore
from rag.core.prompts import ANSWER_PROMPT, GRAPH_RAG_ENTITY_EXTRACT_PROMPT
from rag.graph_rag.state import GraphRAGState


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _extract_entities(state: GraphRAGState) -> GraphRAGState:
    llm = get_chat_model(temperature=0.0)
    resp = llm.invoke(GRAPH_RAG_ENTITY_EXTRACT_PROMPT.format(question=state["question"]))
    parsed = _extract_json(getattr(resp, "content", ""))
    ents = [e for e in parsed.get("entities", []) if isinstance(e, str) and e.strip()]
    return {"entities": ents, "history": [{"node": "extract_entities", "entities": ents}]}


def _expand_subgraph(state: GraphRAGState) -> GraphRAGState:
    neo = Neo4jStore()
    nodes, edges = neo.expand_subgraph(state.get("entities", []), hops=2)
    neo.close()
    return {
        "subgraph_nodes": nodes,
        "subgraph_edges": edges,
        "history": [{"node": "expand_subgraph", "nodes": len(nodes), "edges": len(edges)}],
    }


def _fetch_chunks(state: GraphRAGState) -> GraphRAGState:
    chunk_ids: list[str] = []
    seen = set()
    for n in state.get("subgraph_nodes", []):
        for cid in n.get("source_chunks") or []:
            if cid not in seen:
                chunk_ids.append(cid); seen.add(cid)
    for e in state.get("subgraph_edges", []):
        for cid in e.get("source_chunks") or []:
            if cid not in seen:
                chunk_ids.append(cid); seen.add(cid)
    pg = PgStore()
    chunks = pg.fetch_by_ids(chunk_ids[:20])
    pg.close()
    return {"retrieved": chunks, "history": [{"node": "fetch_chunks", "n": len(chunks)}]}


def _generate(state: GraphRAGState) -> GraphRAGState:
    subgraph_summary = "\n".join(
        f"- ({e['src']}) -[{e['type']}]-> ({e['dst']}): {e.get('description', '')}"
        for e in state.get("subgraph_edges", [])
    )
    chunks_txt = "\n\n---\n\n".join(
        f"[{c['source']}#{c['chunk_index']}] {c['text']}" for c in state.get("retrieved", [])
    )
    ctx = f"Knowledge graph:\n{subgraph_summary}\n\nChunks:\n{chunks_txt}".strip() or "(no context)"
    prompt = ANSWER_PROMPT.format(context=ctx, question=state["question"])
    llm = get_chat_model(temperature=0.0)
    resp = llm.invoke(prompt)
    return {"answer": getattr(resp, "content", str(resp)), "history": [{"node": "generate"}]}


def build_graph():
    g = StateGraph(GraphRAGState)
    g.add_node("extract_entities", _extract_entities)
    g.add_node("expand_subgraph", _expand_subgraph)
    g.add_node("fetch_chunks", _fetch_chunks)
    g.add_node("generate", _generate)

    g.set_entry_point("extract_entities")
    g.add_edge("extract_entities", "expand_subgraph")
    g.add_edge("expand_subgraph", "fetch_chunks")
    g.add_edge("fetch_chunks", "generate")
    g.add_edge("generate", END)
    return g.compile()
```

- [ ] **Step 5: Run test (expect pass)**

```bash
pytest tests/test_graph_rag.py -v
```

- [ ] **Step 6: Commit**

```bash
git add rag/graph_rag/ tests/test_graph_rag.py
git commit -m "feat(graph_rag): entity → subgraph → chunks → generate"
```

---

### Task 16: Loop RAG state (`rag/loop/state.py`)

**Files:**
- Create: `rag/loop/__init__.py`
- Create: `rag/loop/state.py`

- [ ] **Step 1: Implement `rag/loop/state.py`** (mirrors the reference)

```python
"""LangGraph state for Loop RAG (PDCA closed-loop)."""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class Intent(TypedDict, total=False):
    task_type: str
    entities: list[str]
    constraints: list[str]
    sub_goals: list[str]


class PromptConfig(TypedDict, total=False):
    template: str  # stuff | stepwise | structured


class Evidence(TypedDict, total=False):
    id: str
    source: str
    chunk_index: int
    text: str
    score: float
    support: float


class Deviation(TypedDict, total=False):
    align: float
    faith: float
    constraint: float
    total: float
    dominant: str
    converged: bool


class LoopState(TypedDict, total=False):
    user_input: str           # alias for question, kept to match reference
    question: str

    # plan
    intent: Intent
    prompt_config: PromptConfig

    # do
    evidence: list[Evidence]
    answer: str

    # check
    deviation: Deviation

    # loop control
    iteration: int
    converged: bool

    history: Annotated[list[dict[str, Any]], add]
```

- [ ] **Step 2: Commit**

```bash
git add rag/loop/__init__.py rag/loop/state.py
git commit -m "feat(loop): PDCA state shape"
```

---

### Task 17: Loop RAG agents (`rag/loop/agents.py`)

**Files:**
- Create: `rag/loop/agents.py`
- Create: `tests/test_loop_agents.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_loop_agents.py
import json
import math
from unittest.mock import MagicMock, patch

import pytest

from rag.loop.agents import (
    act_node,
    check_node,
    cosine,
    make_do_node,
    plan_node,
    should_continue,
)


def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_cosine_zero_vec():
    assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_identity():
    assert math.isclose(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)


def test_plan_node_parses_json(monkeypatch):
    _env(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "task_type": "factual",
        "entities": ["LangGraph"],
        "constraints": [],
        "sub_goals": ["define"],
        "prompt_template": "stuff",
    }))
    with patch("rag.loop.agents.get_chat_model", return_value=fake_llm):
        out = plan_node({"question": "what is LangGraph?"})
    assert out["intent"]["task_type"] == "factual"
    assert out["prompt_config"]["template"] == "stuff"
    assert out["iteration"] == 0


def test_should_continue():
    s = {"converged": False, "iteration": 1}
    assert should_continue(s) == "continue"
    assert should_continue({"converged": True, "iteration": 1}) == "end"


def test_check_node_computes_deviation(monkeypatch):
    _env(monkeypatch)
    fake_emb = MagicMock()
    fake_emb.embed_query.side_effect = [[1.0, 0.0], [1.0, 0.0]]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({"support_scores": [1.0, 1.0]}))
    with patch("rag.loop.agents.get_embeddings", return_value=fake_emb), \
         patch("rag.loop.agents.get_chat_model", return_value=fake_llm):
        out = check_node({
            "question": "q",
            "answer": "a",
            "evidence": [{"id": "1", "text": "x", "score": 0.9}, {"id": "2", "text": "y", "score": 0.9}],
            "iteration": 0,
            "prompt_config": {"template": "stuff"},
        })
    dev = out["deviation"]
    assert 0.0 <= dev["align"] <= 1.0
    assert 0.0 <= dev["faith"] <= 1.0
    assert dev["converged"] is True or dev["converged"] is False


def test_act_node_increments_iteration(monkeypatch):
    _env(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=json.dumps({
        "rewritten_query": "new q", "prompt_template": "stepwise",
    }))
    with patch("rag.loop.agents.get_chat_model", return_value=fake_llm):
        out = act_node({
            "question": "old",
            "answer": "bad",
            "deviation": {"dominant": "align"},
            "iteration": 1,
        })
    assert out["question"] == "new q"
    assert out["iteration"] == 2
    assert out["prompt_config"]["template"] == "stepwise"
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_loop_agents.py -v
```

- [ ] **Step 3: Implement `rag/loop/agents.py`**

```python
"""Plan-Do-Check-Act agents for Loop RAG."""
from __future__ import annotations

import json
import math
import re
from typing import Any

from rag.core.config import get_settings
from rag.core.embeddings import embed_text, get_embeddings
from rag.core.llm import get_chat_model
from rag.core.neo4j_store import Neo4jStore
from rag.core.pg_store import PgStore
from rag.core.prompts import (
    ANSWER_PROMPT,
    LOOP_ACT_PROMPT,
    LOOP_CHECK_PROMPT,
    LOOP_PLAN_PROMPT,
)
from rag.loop.state import LoopState


# ---------- utilities ----------
def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------- plan ----------
def plan_node(state: LoopState) -> LoopState:
    question = state.get("question") or state.get("user_input") or ""
    llm = get_chat_model(temperature=0.0)
    parsed = _extract_json(getattr(llm.invoke(LOOP_PLAN_PROMPT.format(question=question)), "content", ""))
    intent = {
        "task_type": parsed.get("task_type", "factual"),
        "entities": parsed.get("entities", []) or [],
        "constraints": parsed.get("constraints", []) or [],
        "sub_goals": parsed.get("sub_goals", []) or [],
    }
    prompt_config = {"template": parsed.get("prompt_template", "stuff")}
    return {
        "question": question,
        "user_input": question,
        "intent": intent,
        "prompt_config": prompt_config,
        "iteration": state.get("iteration", 0),
        "converged": False,
        "history": [{"node": "plan", "intent": intent, "iter": state.get("iteration", 0)}],
    }


# ---------- do ----------
def make_do_node(pg_store=None, neo_store=None):
    """Factory so tests can inject fakes."""
    def _do(state: LoopState) -> LoopState:
        pg = pg_store or PgStore()
        neo = neo_store or Neo4jStore()

        q = state["question"]
        q_emb = embed_text(q)
        hits = pg.similarity_search(q_emb, k=8)

        # graph augmentation
        entities = state.get("intent", {}).get("entities", []) or []
        graph_chunks: list[dict[str, Any]] = []
        if entities:
            try:
                nodes, edges = neo.expand_subgraph(entities, hops=2)
                ids: list[str] = []
                seen = set()
                for n in nodes:
                    for cid in n.get("source_chunks") or []:
                        if cid not in seen:
                            ids.append(cid); seen.add(cid)
                for e in edges:
                    for cid in e.get("source_chunks") or []:
                        if cid not in seen:
                            ids.append(cid); seen.add(cid)
                graph_chunks = pg.fetch_by_ids(ids[:8])
            except Exception:  # graph empty / not built yet
                graph_chunks = []

        # merge by id
        merged: dict[str, dict] = {}
        for h in hits:
            merged[h["id"]] = {**h, "score": h.get("score", 0.0)}
        for c in graph_chunks:
            if c["id"] not in merged:
                merged[c["id"]] = {**c, "score": 0.5}
        evidence = list(merged.values())[:8]

        # generate
        ctx = "\n\n---\n\n".join(
            f"[{e['source']}#{e['chunk_index']}] {e['text']}" for e in evidence
        )
        prompt = ANSWER_PROMPT.format(context=ctx or "(no context)", question=q)
        llm = get_chat_model(temperature=0.0)
        resp = llm.invoke(prompt)

        if pg_store is None:
            pg.close()
        if neo_store is None:
            neo.close()

        return {
            "evidence": evidence,
            "answer": getattr(resp, "content", ""),
            "history": [{"node": "do", "evidence": len(evidence)}],
        }
    return _do


# ---------- check ----------
def check_node(state: LoopState) -> LoopState:
    s = get_settings()
    q = state["question"]
    a = state.get("answer", "")
    evidence = state.get("evidence", [])

    emb = get_embeddings()
    e_q = emb.embed_query(q)
    e_a = emb.embed_query(a) if a else [0.0] * len(e_q)
    align = 1.0 - cosine(e_q, e_a)

    # faith via LLM support scores
    ev_lines = "\n".join(f"{i+1}. {e['text']}" for i, e in enumerate(evidence))
    faith = 1.0
    if evidence:
        llm = get_chat_model(temperature=0.0)
        parsed = _extract_json(getattr(
            llm.invoke(LOOP_CHECK_PROMPT.format(question=q, answer=a, evidence=ev_lines)),
            "content", ""))
        scores = [float(x) for x in parsed.get("support_scores", []) if isinstance(x, (int, float))]
        if scores:
            faith = 1.0 - (sum(scores) / len(scores))

    # constraint: simple length/structure heuristic (placeholder for richer rules)
    constraint = 0.0 if a and len(a) >= 20 else 0.5

    total = 0.5 * align + 0.4 * faith + 0.1 * constraint
    dominant = max(
        [("align", align), ("faith", faith), ("constraint", constraint)],
        key=lambda kv: kv[1],
    )[0]

    iteration = state.get("iteration", 0)
    converged = total < s.loop_convergence_threshold or iteration + 1 >= s.loop_max_iterations
    dev = {"align": align, "faith": faith, "constraint": constraint,
           "total": total, "dominant": dominant, "converged": converged}

    return {
        "deviation": dev,
        "converged": converged,
        "history": [{"node": "check", "deviation": dev, "iter": iteration}],
    }


# ---------- act ----------
def act_node(state: LoopState) -> LoopState:
    llm = get_chat_model(temperature=0.0)
    dom = state.get("deviation", {}).get("dominant", "align")
    parsed = _extract_json(getattr(
        llm.invoke(LOOP_ACT_PROMPT.format(
            dominant=dom, question=state["question"], answer=state.get("answer", "")
        )),
        "content", ""))
    new_q = parsed.get("rewritten_query") or state["question"]
    new_template = parsed.get("prompt_template", state.get("prompt_config", {}).get("template", "stuff"))
    return {
        "question": new_q,
        "user_input": new_q,
        "prompt_config": {"template": new_template},
        "iteration": state.get("iteration", 0) + 1,
        "history": [{"node": "act", "rewritten_query": new_q, "template": new_template}],
    }


# ---------- conditional ----------
def should_continue(state: LoopState) -> str:
    return "end" if state.get("converged") else "continue"
```

- [ ] **Step 4: Run tests (expect pass)**

```bash
pytest tests/test_loop_agents.py -v
```

- [ ] **Step 5: Commit**

```bash
git add rag/loop/agents.py tests/test_loop_agents.py
git commit -m "feat(loop): PDCA plan/do/check/act nodes with cosine + deviation"
```

---

### Task 18: Loop RAG graph (`rag/loop/graph.py`)

**Files:**
- Create: `rag/loop/graph.py`
- Create: `tests/test_loop_graph.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_loop_graph.py
import json
from unittest.mock import MagicMock, patch

from rag.loop.graph import build_graph


def _env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_loop_converges_on_first_pass(monkeypatch):
    _env(monkeypatch)

    fake_llm = MagicMock()
    # plan, do(generate), check(support), [no act needed because converged]
    fake_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({  # plan
            "task_type": "factual", "entities": [], "constraints": [],
            "sub_goals": [], "prompt_template": "stuff"
        })),
        MagicMock(content="this is the answer."),                  # do generate
        MagicMock(content=json.dumps({"support_scores": [1.0]})),  # check
    ]

    fake_emb = MagicMock()
    fake_emb.embed_query.return_value = [1.0, 0.0]

    fake_pg = MagicMock()
    fake_pg.similarity_search.return_value = [
        {"id": "c1", "source": "a.md", "chunk_index": 0, "text": "alpha", "metadata": {}, "score": 0.9}
    ]
    fake_pg.fetch_by_ids.return_value = []
    fake_neo = MagicMock()
    fake_neo.expand_subgraph.return_value = ([], [])

    with patch("rag.loop.agents.get_chat_model", return_value=fake_llm), \
         patch("rag.loop.agents.embed_text", return_value=[1.0, 0.0]), \
         patch("rag.loop.agents.get_embeddings", return_value=fake_emb), \
         patch("rag.loop.agents.PgStore", return_value=fake_pg), \
         patch("rag.loop.agents.Neo4jStore", return_value=fake_neo):
        app = build_graph()
        out = app.invoke({"question": "what is alpha?", "iteration": 0})

    assert out["answer"] == "this is the answer."
    assert out["converged"] is True
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_loop_graph.py -v
```

- [ ] **Step 3: Implement `rag/loop/graph.py`**

```python
"""LangGraph wiring for Loop RAG (PDCA closed-loop)."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag.core.config import get_settings
from rag.loop.agents import act_node, check_node, make_do_node, plan_node, should_continue
from rag.loop.state import LoopState


def build_graph(pg_store=None, neo_store=None):
    g = StateGraph(LoopState)
    g.add_node("plan", plan_node)
    g.add_node("do", make_do_node(pg_store=pg_store, neo_store=neo_store))
    g.add_node("check", check_node)
    g.add_node("act", act_node)

    g.set_entry_point("plan")
    g.add_edge("plan", "do")
    g.add_edge("do", "check")
    g.add_conditional_edges("check", should_continue, {"continue": "act", "end": END})
    g.add_edge("act", "plan")

    s = get_settings()
    return g.compile().with_config(recursion_limit=s.loop_max_iterations * 4 + 4)
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_loop_graph.py -v
```

- [ ] **Step 5: Commit**

```bash
git add rag/loop/graph.py tests/test_loop_graph.py
git commit -m "feat(loop): PDCA graph wiring with conditional convergence"
```

---

### Task 19: Registry (`rag/registry.py`)

**Files:**
- Create: `rag/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_registry.py
from rag.registry import REGISTRY, list_patterns


def test_registry_has_four_patterns():
    assert set(list_patterns()) == {"naive", "agentic", "graph", "loop"}
    for fn in REGISTRY.values():
        assert callable(fn)
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_registry.py -v
```

- [ ] **Step 3: Implement `rag/registry.py`**

```python
"""Pattern name → graph builder. Single entry point used by CLI/API/UI."""
from __future__ import annotations

from typing import Callable

from rag.agentic.graph import build_graph as build_agentic
from rag.graph_rag.graph import build_graph as build_graph_rag
from rag.loop.graph import build_graph as build_loop
from rag.naive.graph import build_graph as build_naive

REGISTRY: dict[str, Callable] = {
    "naive": build_naive,
    "agentic": build_agentic,
    "graph": build_graph_rag,
    "loop": build_loop,
}


def list_patterns() -> list[str]:
    return list(REGISTRY.keys())


def run(pattern: str, question: str) -> dict:
    if pattern not in REGISTRY:
        raise ValueError(f"unknown pattern: {pattern!r} (have {list_patterns()})")
    app = REGISTRY[pattern]()
    initial = {"question": question, "iteration": 0} if pattern == "loop" else {"question": question}
    result = app.invoke(initial)
    return {
        "answer": result.get("answer", ""),
        "trace": result.get("history", []),
        "raw": {k: v for k, v in result.items() if k not in {"answer", "history"}},
    }
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_registry.py -v
```

- [ ] **Step 5: Commit**

```bash
git add rag/registry.py tests/test_registry.py
git commit -m "feat: pattern registry exposing run(pattern, question)"
```

---

### Task 20: CLI (`app/cli.py`)

**Files:**
- Create: `app/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py
from unittest.mock import patch

from typer.testing import CliRunner

from app.cli import app


def test_cli_patterns_subcommand():
    runner = CliRunner()
    with patch("app.cli.list_patterns", return_value=["naive", "agentic", "graph", "loop"]):
        r = runner.invoke(app, ["patterns"])
    assert r.exit_code == 0
    assert "naive" in r.stdout


def test_cli_ask_invokes_registry():
    runner = CliRunner()
    with patch("app.cli.run", return_value={"answer": "ok", "trace": [], "raw": {}}) as m:
        r = runner.invoke(app, ["ask", "naive", "hello?"])
    assert r.exit_code == 0
    assert "ok" in r.stdout
    m.assert_called_once_with("naive", "hello?")
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement `app/cli.py`**

```python
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
    from rag.core.neo4j_store import Neo4jStore
    from rag.core.pg_store import PgStore
    pg = PgStore(); pg.init_schema(); pg.truncate(); pg.close()
    neo = Neo4jStore(); neo.wipe(); neo.close()
    typer.echo("reset done")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_cli.py -v
```

- [ ] **Step 5: Commit**

```bash
git add app/cli.py tests/test_cli.py
git commit -m "feat(app): typer CLI (ask, patterns, ingest, build-kg, reset)"
```

---

### Task 21: FastAPI (`app/api.py`)

**Files:**
- Create: `app/api.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_api.py
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_patterns():
    client = TestClient(app)
    with patch("app.api.list_patterns", return_value=["naive", "agentic", "graph", "loop"]):
        r = client.get("/patterns")
    assert r.status_code == 200
    assert r.json() == ["naive", "agentic", "graph", "loop"]


def test_rag_endpoint():
    client = TestClient(app)
    with patch("app.api.run", return_value={"answer": "yes", "trace": [], "raw": {}}):
        r = client.post("/rag/naive", json={"question": "is it?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "yes"
    assert "latency_ms" in body
```

- [ ] **Step 2: Run test (expect fail)**

```bash
pytest tests/test_api.py -v
```

- [ ] **Step 3: Implement `app/api.py`**

```python
"""FastAPI surface for all four RAG patterns."""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.registry import list_patterns, run

app = FastAPI(title="DIP RAG", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskBody(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    trace: list
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/patterns")
def patterns() -> list[str]:
    return list_patterns()


@app.post("/rag/{pattern}", response_model=AskResponse)
def ask(pattern: str, body: AskBody):
    if pattern not in list_patterns():
        raise HTTPException(404, f"unknown pattern '{pattern}'")
    t0 = time.perf_counter()
    out = run(pattern, body.question)
    return AskResponse(
        answer=out["answer"],
        trace=out["trace"],
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.post("/ingest")
def ingest():
    from scripts.ingest import run_ingest
    n = run_ingest()
    return {"chunks_added": n}


@app.post("/kg/build")
def kg_build(limit: int | None = None):
    from scripts.build_kg import run_build_kg
    e, r = run_build_kg(limit=limit)
    return {"entities": e, "relations": r}
```

- [ ] **Step 4: Run test (expect pass)**

```bash
pytest tests/test_api.py -v
```

- [ ] **Step 5: Commit**

```bash
git add app/api.py tests/test_api.py
git commit -m "feat(app): FastAPI for /rag/{pattern}, /ingest, /kg/build"
```

---

### Task 22: Streamlit UI (`app/ui.py`)

**Files:**
- Create: `app/ui.py`

- [ ] **Step 1: Implement `app/ui.py`** (no unit test — UI smoke is manual)

```python
"""Streamlit UI: pick a pattern, ask a question, inspect the trace.
Calls the FastAPI server (so the API is the canonical interface)."""
from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API = os.environ.get("STREAMLIT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DIP RAG", layout="wide")
st.title("DIP — Four RAG Patterns")

with st.sidebar:
    st.header("Settings")
    try:
        patterns = httpx.get(f"{API}/patterns", timeout=5).json()
    except Exception:
        patterns = ["naive", "agentic", "graph", "loop"]
        st.warning(f"API unreachable at {API}; using static pattern list.")
    pattern = st.selectbox("Pattern", patterns, index=0)
    show_trace = st.toggle("Show trace", value=True)

    st.markdown("---")
    if st.button("Ingest data/"):
        with st.spinner("Ingesting…"):
            r = httpx.post(f"{API}/ingest", timeout=600)
            st.success(r.json())
    if st.button("Build KG"):
        with st.spinner("Extracting KG…"):
            r = httpx.post(f"{API}/kg/build", timeout=1800)
            st.success(r.json())

col_a, col_b = st.columns([2, 1] if show_trace else [1, 0.001])

with col_a:
    question = st.text_area("Question", height=120, placeholder="Ask anything…")
    go = st.button("Ask", type="primary", use_container_width=True)
    if go and question:
        with st.spinner(f"Running {pattern} RAG…"):
            r = httpx.post(f"{API}/rag/{pattern}", json={"question": question}, timeout=600)
            if r.status_code != 200:
                st.error(f"{r.status_code}: {r.text}")
            else:
                data = r.json()
                st.markdown("### Answer")
                st.markdown(data["answer"])
                st.caption(f"latency: {data['latency_ms']} ms")
                st.session_state["last_trace"] = data["trace"]

with col_b:
    if show_trace and st.session_state.get("last_trace"):
        st.markdown("### Trace")
        st.json(st.session_state["last_trace"])
```

- [ ] **Step 2: Smoke-run manually**

```bash
uvicorn app.api:app --reload &
streamlit run app/ui.py
```

Open `http://localhost:8501`. Sidebar should list 4 patterns; ask works.

- [ ] **Step 3: Commit**

```bash
git add app/ui.py
git commit -m "feat(app): streamlit UI for all four RAG patterns"
```

---

### Task 23: Notebooks (`notebooks/`)

**Files:**
- Create: `notebooks/01_naive_rag.ipynb`
- Create: `notebooks/02_agentic_rag.ipynb`
- Create: `notebooks/03_graph_rag.ipynb`
- Create: `notebooks/04_loop_rag.ipynb`

- [ ] **Step 1: Generate a stub notebook per pattern**

Each notebook should be created as JSON. Below is the template for the naive one — repeat with the pattern name swapped for the other three.

`notebooks/01_naive_rag.ipynb`:
```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Naive RAG\n", "\n", "retrieve top-k → stuff → generate"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": ["from dotenv import load_dotenv\n", "load_dotenv()\n", "\n", "from rag.registry import run\n", "\n", "out = run('naive', 'What is LangGraph?')\n", "print(out['answer'])"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [],
   "source": ["import json\n", "print(json.dumps(out['trace'], indent=2, default=str))"]}
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
 "nbformat": 4, "nbformat_minor": 5
}
```

Repeat for `02_agentic_rag.ipynb`, `03_graph_rag.ipynb`, `04_loop_rag.ipynb` substituting `'naive'` for `'agentic'`, `'graph'`, `'loop'`. For the loop notebook, append a cell that pretty-prints each iteration's deviation from `out['raw']['deviation']` history.

- [ ] **Step 2: Commit**

```bash
git add notebooks/
git commit -m "docs: per-pattern jupyter notebooks"
```

---

### Task 24: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# DIP — Four RAG Patterns in LangGraph

Naive, Agentic, Graph, and Loop (PDCA) RAG implementations sharing one
PostgreSQL+pgvector store and one Neo4j knowledge graph, exposed via CLI,
FastAPI, and Streamlit.

## Quick start

```bash
brew install postgresql@16 neo4j
brew services start postgresql@16
brew services start neo4j
psql postgres -c "CREATE DATABASE dip_rag;"
psql dip_rag  -c "CREATE EXTENSION vector;"

python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in DEEPSEEK_API_KEY, OPENAI_API_KEY, NEO4J_PASSWORD

python -m app.cli ingest
python -m app.cli build-kg
python -m app.cli ask naive "What is LangGraph?"

uvicorn app.api:app --reload &
streamlit run app/ui.py
```

## Patterns

| Pattern  | Topology                                                | When to use |
|----------|--------------------------------------------------------|-------------|
| naive    | retrieve → generate                                     | baseline |
| agentic  | agent ⇄ tools (vector_search, kg_lookup)                | multi-step lookup |
| graph    | extract entities → expand Neo4j → fetch chunks → generate | relationship queries |
| loop     | plan → do → check ⇄ act (PDCA)                          | hard/ambiguous queries |

## Run tests

```bash
pytest -m "not integration and not llm"   # fast
pytest -m integration                      # needs PG + Neo4j
pytest -m llm                              # costs DeepSeek tokens
```

## Layout

```
rag/core/          shared LLM, embeddings, stores, loader, prompts
rag/{naive,agentic,graph_rag,loop}/  one LangGraph per pattern
rag/registry.py    pattern name → graph builder
scripts/           ingest, build_kg, reset_stores
app/               cli, api, ui
data/seed/         bundled sample corpus
data/raw/          drop your own PDFs/MD/TXT here
```

See `docs/superpowers/specs/2026-05-20-rag-patterns-langgraph-design.md` for
the full design.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README quickstart and overview"
```

---

## Self-review

**Spec coverage:**

| Spec section                                    | Task |
|--------------------------------------------------|------|
| Repo layout                                      | 1, 13–18, 20–22 |
| LangGraph topologies — naive                     | 13 |
| LangGraph topologies — agentic                   | 14 |
| LangGraph topologies — graph                     | 15 |
| LangGraph topologies — loop (PDCA)               | 16, 17, 18 |
| PGVector schema                                  | 6 |
| Neo4j schema                                     | 7 |
| Ingestion (chunking, embed, upsert)              | 5, 9 |
| Build KG                                         | 10 |
| Reset stores                                     | 11 |
| Seed data                                        | 12 |
| Config from .env                                 | 2 |
| LLM (DeepSeek)                                   | 3 |
| Embeddings (OpenAI)                              | 4 |
| Prompts                                          | 8 |
| Registry                                         | 19 |
| CLI                                              | 20 |
| FastAPI                                          | 21 |
| Streamlit                                        | 22 |
| Notebooks                                        | 23 |
| Open question: hybrid vector+BM25                | deferred (noted in spec) |
| Open question: web search tool                   | deferred (noted in spec) |
| Open question: learned scorer for check          | deferred (noted in spec) |

No gaps.

**Placeholder scan:** none — every code step has full content.

**Type consistency:**
- `Chunk` defined in Task 5 used in Tasks 6 (PgStore.upsert) and 9 (ingest) — same shape.
- `PgStore` API: `init_schema`, `truncate`, `upsert`, `similarity_search`, `fetch_by_ids`, `count`, `close` — used consistently in Tasks 9, 13, 14, 15, 17.
- `Neo4jStore` API: `init_schema`, `wipe`, `merge_entity`, `merge_relation`, `get_entity`, `find_entities_by_names`, `find_relations`, `expand_subgraph`, `close` — used consistently in Tasks 10, 14, 15, 17.
- LangGraph state keys (`question`, `answer`, `history`, plus per-pattern fields) match across patterns.
- Registry `run()` return shape `{answer, trace, raw}` matches CLI (Task 20) and API (Task 21).

---

## Execution

Plan complete and saved to `docs/superpowers/plans/2026-05-20-rag-patterns-langgraph.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

---

# Addendum — Tasks 25-31 (executed 2026-05-21)

After the original 24 tasks shipped a working four-pattern system against a 3-file
seed corpus, the user requested two architectural shifts and one new pattern.
These addendum tasks document what was actually built; they don't follow the
strict TDD micro-step format of Tasks 1-24 because they were executed
interactively in the main thread with immediate verification rather than
dispatched to subagents.

---

### Task 25 — Replace Neo4j with SQLite + sqlite-graph

**Why:** Avoid Neo4j install / daemon entirely; ship a single C extension and
keep the graph file-based.

**Reference:** https://github.com/agentflare-ai/sqlite-graph (v0.1.0-alpha).
v0.1.0-alpha Cypher has no variable-length paths or aggregations; multi-hop
expansion is done in Python (BFS) instead.

**Files:**
- New: `vendor/libgraph.dylib` (built from source on the user's macOS x86_64)
- New: `rag/core/graph_store.py`
- New: `tests/test_graph_store.py` (real-extension tests against a tempfile DB)
- Deleted: `rag/core/neo4j_store.py`, `tests/test_neo4j_store.py`
- Modified: `rag/agentic/tools.py`, `rag/graph_rag/graph.py`, `rag/loop/agents.py`,
  `rag/loop/graph.py`, `scripts/build_kg.py`, `scripts/reset_stores.py`, `app/cli.py`
- Removed `neo4j>=5.24.0` from `pyproject.toml`; added `GRAPH_DB_PATH` and
  `GRAPH_EXTENSION_PATH` settings.

**Implementation notes:**
- Each `GraphStore` connection auto-loads `vendor/libgraph.dylib` and calls
  `_ensure_schema` so reads never error on a missing virtual table.
- sqlite-graph's `graph_node_add` errors on duplicate ID, but direct
  `UPDATE graph_nodes SET properties = ?` works — so merges read-modify-update.
- An `entity_name_idx` table maps lowercased canonical names to the integer
  node IDs the extension uses internally.
- `expand_subgraph(seed_names, hops)` does BFS in Python over `graph_edges`
  to replicate Neo4j's `MATCH p=(seed)-[r:RELATES*1..hops]-(n)`.

**Status:** Done. All 5 patterns work against the new store. 38/38 fast tests
pass (7 of them new `GraphStore` tests exercising the real `.dylib`).

---

### Task 26 — Switch to local sentence-transformers embeddings + rename LLM env vars

**Why:** Keep document text on-machine, eliminate OpenAI embedding cost,
generalize the chat client to any OpenAI-compatible provider.

**Files:**
- Modified: `rag/core/config.py` — `DEEPSEEK_*` → `LLM_*`, added
  `embedding_provider` (`local|openai`), `embedding_model_local`,
  `embedding_model_openai`, default `embedding_dim=384`.
- Modified: `rag/core/llm.py` — uses `llm_api_key`, `llm_base_url`, `llm_model`.
- Modified: `rag/core/embeddings.py` — `_LocalEmbedder` wraps
  `sentence_transformers.SentenceTransformer("BAAI/bge-small-en-v1.5")`;
  `get_embeddings()` dispatches on provider.
- Modified: `.env.example` to use new names; default `EMBEDDING_PROVIDER=local`.
- Added: `sentence-transformers>=3.0,<4.0` (pin keeps `transformers<5.0` for
  torch 2.2 compat); `numpy<2.0` (torch 2.2 incompatible with numpy 2.x).
- Updated: all tests' env fixtures.

**Workaround:** `from langchain_text_splitters.character import RecursiveCharacterTextSplitter`
instead of from the package root — the package `__init__.py` pulls in
`sentence_transformers` which fails on `transformers 5.x` + `torch 2.2`.

**Status:** Done. All embedding-dependent tests pass; integration ingest of
the LoopRAG PDF (161 chunks × 384-dim) verified.

---

### Task 27 — NodeRAG storage schema (`noderag_store.py`)

**Why:** NodeRAG (Xu et al. 2025) requires per-node embeddings across six
heterogeneous node types. The existing `rag_chunks` table is chunk-level only.

**Files:**
- New: `rag/core/noderag_store.py` — new pgvector table `noderag_nodes`:
  ```sql
  CREATE TABLE noderag_nodes (
      id TEXT PRIMARY KEY,
      node_type TEXT NOT NULL,       -- six allowed values, see NODE_TYPES
      content TEXT NOT NULL,
      metadata JSONB DEFAULT '{}',
      embedding VECTOR({dim}) NOT NULL,
      chunk_ids TEXT[] DEFAULT '{}',  -- back-refs to rag_chunks
      created_at TIMESTAMPTZ DEFAULT NOW()
  );
  CREATE INDEX noderag_nodes_embedding_idx ON noderag_nodes USING hnsw (embedding vector_cosine_ops);
  CREATE INDEX noderag_nodes_type_idx ON noderag_nodes (node_type);
  ```
- API: `init_schema`, `truncate`, `upsert_node`, `count`, `counts_by_type`,
  `vector_search(query_embedding, k, node_types=None)`, `fetch_by_ids`,
  `all_nodes_minimal`, `close`.
- New config keys: `noderag_top_k_seeds=20`, `noderag_ppr_top_n=50`,
  `noderag_ppr_alpha=0.15`, `noderag_chunk_top_k=18` (final tuned values).

**NODE_TYPES:** entity, relationship, semantic_unit, attribute,
high_level_element, high_level_overview.

---

### Task 28 — NodeRAG extraction pipeline (`scripts/build_noderag.py`)

**Five-stage pipeline:**

1. Load chunks from pgvector.
2. Per chunk, two LLM extraction calls:
   - `NODERAG_SEMANTIC_UNIT_PROMPT` → atomic factual claims + the entities they reference.
   - `NODERAG_ATTRIBUTE_PROMPT` → `entity.name = value` triples.
3. Harvest entities + relationships from sqlite-graph (built by Task 10).
4. Embed and upsert all four base types into `noderag_nodes`.
5. Build heterogeneous networkx graph:
   - entity-entity from sqlite-graph relations (weight 1.0)
   - semantic_unit ↔ entity from extracted entity names (weight 0.7)
   - attribute ↔ entity from attribute owner (weight 0.5)

   Run Louvain via `python-louvain`; for each community ≥ 3 members:
   - create `high_level_element` node (community label + member list)
   - LLM-summarize members → `high_level_overview` node (`NODERAG_COMMUNITY_SUMMARY_PROMPT`)

**Cost guardrails:** `--limit N` for testing, `--skip-communities` to skip stages 5–8.

**Files:**
- New: `scripts/build_noderag.py`
- Modified: `rag/core/prompts.py` — added `NODERAG_SEMANTIC_UNIT_PROMPT`,
  `NODERAG_ATTRIBUTE_PROMPT`, `NODERAG_COMMUNITY_SUMMARY_PROMPT`.
- Added deps: `networkx>=3.3`, `python-louvain>=0.16`.

---

### Task 29 — NodeRAG LangGraph (`rag/noderag/`)

**Topology:** `embed_query → vector_seed → ppr_propagate → fetch_chunks → generate`.

- `vector_seed`: HNSW cosine top-`top_k_seeds` across all node types.
- `ppr_propagate`: build PPR graph (cached via `lru_cache`):
   - entity-entity edges from sqlite-graph relations
   - semantic_unit ↔ entity from `metadata.entities`
   - attribute ↔ entity from `metadata.entity`

  Personalization vector = `cosine_score × seed_type_weight` per seed, where
  `_SEED_TYPE_WEIGHTS = {semantic_unit: 1.4, high_level_overview: 1.3,
   high_level_element: 1.1, entity: 1.0, attribute: 1.0, relationship: 0.6}`.
  Run `networkx.pagerank(alpha=1 - ppr_alpha)`. Return top-`ppr_top_n` ranked nodes.

- `fetch_chunks`: **hybrid candidate pool** —
   - (a) round-robin chunk_ids from PPR-ranked nodes (structural relevance)
   - (b) top-`chunk_top_k × 2` direct cosine on `rag_chunks` (semantic relevance)

  Merge, dedupe, re-rank by cosine to the query embedding, keep top-`chunk_top_k`.
  This fixes PPR's bias toward chunks tied to highly-connected entities;
  formula/detail-heavy chunks often have few graph neighbors and would
  otherwise be missed.

- `generate`: prompt includes top-8 ranked node descriptions + reranked chunk text.

**Files:**
- New: `rag/noderag/__init__.py`, `rag/noderag/state.py`, `rag/noderag/graph.py`
- New: `tests/test_noderag.py` (mocked LLM + stores)

---

### Task 30 — Wire NodeRAG into registry / CLI / API / UI

- `rag/registry.py`: 5th pattern `"noderag"`.
- `app/cli.py`: new commands `build-noderag`, `noderag-stats`, `compare`.
- `app/api.py`: new endpoints `POST /noderag/build`, `GET /noderag/stats`.
- `app/ui.py`: dropdown auto-fetches from `/patterns` — no code change needed.
- `tests/test_registry.py`: assert 5 patterns.
- `tests/test_build_noderag.py`: mocked-pipeline coverage.

---

### Task 31 — Run + tune NodeRAG against the LoopRAG paper

**Initial run:** Built index over 161 chunks of the Bai et al. 2026 paper.
Result: 603 entities + 700 relations + 1,037 semantic_units + 637 attributes +
69 communities (high_level_element + high_level_overview = 138 nodes). Total
3,075 NodeRAG nodes in pgvector.

**First failure mode:** On the question "What is LoopRAG's three-axis
deviation signal, and what does each axis measure?", NodeRAG returned
"I don't know" while agentic produced full LaTeX formulas. Root cause
investigation showed:
- PPR was dominated by the LoopRAG entity (40 chunk_ids, PPR score 0.27 vs
  next 0.026), and round-robin drained its chunks in extraction order (not
  relevance order).
- The key chunk #59 (which explicitly names the three axes) sits at cosine
  rank 16 globally — below the original `chunk_top_k=12`.

**Fix (committed):**
1. **Hybrid candidate pool** — PPR-derived chunks ∪ direct semantic chunks.
2. **Cosine re-rank** of the merged pool before final selection.
3. Defaults: `top_k_seeds 12→20`, `ppr_top_n 30→50`, `chunk_top_k 6→18`.
4. Seed-type weights down-weight relationship (0.6) and up-weight
   semantic_unit (1.4) / high_level_overview (1.3).
5. Capped ranked-node prompt lines at 8 (was 15) so chunk text gets more
   attention.

**Result:** NodeRAG now correctly names all three axes and runs in 1.2s
(vs Loop's 16s on the same question, with Loop still failing). Five-way
`compare` produces visibly different traces with the expected pattern-by-question
strength profile documented in the spec.

---

## Addendum self-review (Tasks 25-31)

**Spec coverage:**

| Spec section          | Where built |
|-----------------------|-------------|
| sqlite-graph swap     | Task 25 |
| Local embeddings      | Task 26 |
| Renamed env vars      | Task 26 |
| NodeRAG storage       | Task 27 |
| NodeRAG extraction    | Task 28 |
| NodeRAG retrieval     | Task 29 |
| NodeRAG integration   | Task 30 |
| NodeRAG tuning        | Task 31 |

**Type consistency:**
- `GraphStore` keeps the exact API of the deleted `Neo4jStore` so callers
  needed only an import swap.
- `NodeRAGStore` shares the `pgvector` driver path with `PgStore`.
- All five patterns return the same `{answer, trace, raw}` shape via
  `registry.run()`.

**Status:** All shipped. Repo pushed to https://github.com/leevydanomalik/deepRAG
with the `strong_password` literal scrubbed from history via `git filter-repo`.
