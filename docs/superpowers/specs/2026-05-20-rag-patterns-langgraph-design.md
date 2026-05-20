# deepRAG — Five RAG Patterns in LangGraph (Design)

**Status:** Living document — reflects shipped state as of 2026-05-21
**Initial draft:** 2026-05-20
**Owner:** Leevyd Malik

## Goal

Build a single Python project that implements **five RAG patterns as LangGraph
graphs** sharing one ingestion pipeline, one vector store (PGVector), and one
heterogeneous graph (SQLite + sqlite-graph extension). Expose all five through
a CLI, a FastAPI server, and a Streamlit UI so they can be compared
interactively against the same corpus.

The five patterns:

1. **Naive RAG** — retrieve once, generate.
2. **Agentic RAG** — LLM tool-calls a retriever / KG lookup until it has enough context.
3. **Graph RAG** — entity extraction + 2-hop subgraph traversal (BFS over sqlite-graph) + chunk lookup.
4. **Loop RAG** — PDCA (Plan-Do-Check-Act) closed-loop multi-agent, iterating
   until a three-axis deviation signal converges. Mirrors the pattern in
   `/Users/leevydmalik/Documents/Project/DAYA/LOOPRAG`.
5. **NodeRAG** — Xu et al. 2025 heterogeneous-node retrieval. Six node types
   (entity, relationship, semantic_unit, attribute, high_level_element,
   high_level_overview) with HNSW vector seeds + Personalized PageRank
   propagation + hybrid candidate pool with cosine re-rank.

## Non-goals

- Production deployment (auth, rate limiting, multi-tenant) — single user, local.
- Evaluation harness (RAGAS, etc.) — out of scope for this milestone.
- Streaming responses — first cut is request/response only.

## Stack

| Concern          | Choice |
|------------------|--------|
| LLM (chat)       | DeepSeek `deepseek-chat` via OpenAI-compatible API |
| Embeddings       | Local `BAAI/bge-small-en-v1.5` via sentence-transformers (384-dim); OpenAI `text-embedding-3-small` (1536-dim) selectable via `EMBEDDING_PROVIDER` |
| Vector store     | PostgreSQL + `pgvector` — two tables: `rag_chunks` (text chunks) and `noderag_nodes` (heterogeneous nodes) |
| Graph store      | SQLite + [sqlite-graph](https://github.com/agentflare-ai/sqlite-graph) C extension (`vendor/libgraph.dylib`, vendored). Multi-hop expansion done in Python (BFS) because v0.1.0-alpha Cypher has no variable-length paths. |
| Graph algorithms | networkx (Personalized PageRank), python-louvain (community detection) |
| Orchestration    | LangGraph 0.2 |
| CLI              | Typer |
| API              | FastAPI + uvicorn |
| UI               | Streamlit |
| Python           | 3.11+ (tested on 3.12) |

The choice of SQLite + sqlite-graph over Neo4j is deliberate: zero infrastructure beyond a 328 KB
vendored `.dylib`, no daemon, fully embeddable, file-based persistence. The alpha-stage Cypher
limitations are worked around by doing multi-hop traversal and PPR directly in Python.

## Repository layout

```
deepRAG/
├── pyproject.toml
├── .env.example
├── README.md
├── vendor/
│   └── libgraph.dylib              # sqlite-graph C extension (macOS x86_64)
├── data/
│   ├── seed/                       # bundled sample markdown
│   ├── raw/                        # user drops PDFs/MD/TXT here
│   └── graph.db                    # sqlite-graph DB (gitignored)
├── scripts/
│   ├── ingest.py                   # chunk → embed → upsert to pgvector (rag_chunks)
│   ├── build_kg.py                 # LLM-extract entities/relations → sqlite-graph
│   ├── build_noderag.py            # NodeRAG pipeline (semantic_units, attributes, communities)
│   └── reset_stores.py             # truncate pgvector + wipe sqlite-graph (--confirm)
├── rag/
│   ├── core/
│   │   ├── llm.py                  # DeepSeek (OpenAI-compatible) chat client
│   │   ├── embeddings.py           # local sentence-transformers OR OpenAI
│   │   ├── pg_store.py             # pgvector retrieval over rag_chunks
│   │   ├── graph_store.py          # sqlite-graph adapter (entities + relations)
│   │   ├── noderag_store.py        # pgvector adapter for noderag_nodes (6 node types)
│   │   ├── loader.py               # PDF/MD/TXT loaders + chunking (NUL-strip on PDF)
│   │   ├── prompts.py              # ANSWER, KG_EXTRACTION, LOOP_*, NODERAG_* prompts
│   │   └── config.py               # pydantic-settings, loads .env
│   ├── naive/         {state,graph}.py
│   ├── agentic/       {state,tools,graph}.py     # vector_search, kg_lookup
│   ├── graph_rag/     {state,graph}.py
│   ├── loop/          {state,agents,graph}.py    # PDCA
│   ├── noderag/       {state,graph}.py           # NodeRAG (HNSW + PPR)
│   └── registry.py                 # pattern name → build_graph fn (5 patterns)
├── app/
│   ├── cli.py                      # ask, compare, ingest, build-kg, build-noderag, noderag-stats, reset
│   ├── api.py                      # FastAPI: POST /rag/{pattern}, /ingest, /kg/build, /noderag/build, /noderag/stats
│   └── ui.py                       # Streamlit dropdown + chat + trace panel
├── tests/                          # 39+ unit tests; integration tests for pg_store / graph_store
└── notebooks/
    ├── 01_naive_rag.ipynb
    ├── 02_agentic_rag.ipynb
    ├── 03_graph_rag.ipynb
    └── 04_loop_rag.ipynb
```

Each pattern is a self-contained LangGraph exposing the same interface:

```python
def build_graph() -> CompiledStateGraph: ...
# input:  {"question": str}
# output: {"answer": str, "history": list[dict]}
```

## LangGraph topologies

### Naive RAG

```
START → retrieve → generate → END
```

- `retrieve`: top-k cosine similarity against `rag_chunks`, k=5.
- `generate`: stuff prompt, single DeepSeek call.

### Agentic RAG

```
START → agent ⇄ tools
            │
            └─(no tool calls)→ END
```

- Built with LangGraph's prebuilt ReAct/tool-calling pattern.
- Tools: `vector_search(query, k=5)`, `kg_lookup(entity_name)`.
- Stop condition: agent emits a message with no tool calls.
- `recursion_limit` cap = 8 to prevent runaway loops.

### Graph RAG

```
START → extract_query_entities → expand_subgraph → fetch_chunks → generate → END
```

- `extract_query_entities`: DeepSeek extracts entity mentions from the question.
- `expand_subgraph`: **2-hop BFS in Python** over the sqlite-graph DB (no Cypher
  variable-length paths in v0.1.0-alpha). Returns matched nodes + edges.
- `fetch_chunks`: collect `source_chunks` IDs from matched entities/relations, then fetch
  full text from `rag_chunks` via `PgStore.fetch_by_ids`.
- `generate`: stuff prompt with subgraph summary + chunk text.

### Loop RAG (PDCA)

```
            ┌──── plan ─────┐
            │               ▼
            │              do
            │               │
            │               ▼
            │            check ── converged? ── END
            │               │ no
            │               ▼
            └────────────── act
```

- **plan**: classify task type (factual | explanatory | comparative | predictive |
  control | diagnosis), extract entities + constraints, pick prompt template.
- **do**: hybrid retrieve (pgvector cosine + sqlite-graph subgraph from plan entities) →
  generate answer. Falls back gracefully (vector-only) if the graph DB is empty.
- **check**: compute three-axis deviation
  - `align = 1 − cos(E(answer), E(question))`
  - `faith = 1 − mean(δᵢ)` where δᵢ is per-evidence support score (LLM-graded)
  - `constraint = 1 − Ψ(answer, prompt_policy)` (length / structural heuristic)
  - `total = 0.5·align + 0.4·faith + 0.1·constraint`
  - converged when `total < LOOP_CONVERGENCE_THRESHOLD` or `iteration >= LOOP_MAX_ITERATIONS`.
- **act**: rewrite intent + swap template based on dominant deviation axis.

State is the same shape as the reference `LoopRAGState` (intent, task_graph,
prompt_config, evidence, answer, deviation, iteration, converged, history).

### NodeRAG (Xu et al. 2025)

```
START → embed_query → vector_seed → ppr_propagate → fetch_chunks → generate → END
```

Six node types live in `noderag_nodes` (pgvector), each with its own embedding:

| Node type            | Created from |
|----------------------|--------------|
| `entity`             | sqlite-graph entities (harvested from build-kg) |
| `relationship`       | sqlite-graph edges (promoted to first-class nodes) |
| `semantic_unit`      | LLM extraction per chunk — atomic factual claims |
| `attribute`          | LLM extraction per chunk — entity.name = value |
| `high_level_element` | Louvain community labels over the heterogeneous graph |
| `high_level_overview`| LLM-written one-paragraph summary per community |

Retrieval pipeline:

- **embed_query**: embed question once (cached on state).
- **vector_seed**: HNSW cosine search across `noderag_nodes` — top `top_k_seeds`
  (default 20) across all six node types.
- **ppr_propagate**: build the heterogeneous graph (cached) — entity-entity from
  sqlite-graph relations; semantic_unit ↔ entity from extracted entity names;
  attribute ↔ entity from attribute owner. Personalized PageRank with seeds
  weighted by `(cosine_score × type_weight)` — semantic_unit (1.4×) and
  high_level_overview (1.3×) up-weighted; relationship (0.6×) down-weighted
  since relationships rarely carry full factual text. Returns top `ppr_top_n`
  (default 50) ranked nodes.
- **fetch_chunks**: hybrid candidate pool combining
  - (a) round-robin chunks from PPR-ranked nodes (structural relevance)
  - (b) top-K direct cosine search on `rag_chunks` (semantic relevance)
  
  then re-rank merged pool by cosine against query embedding and keep top
  `chunk_top_k` (default 18). The hybrid pool fixes PPR's structural bias
  toward chunks tied to highly-connected entities; the rerank prevents
  drainage of any one entity's chunks dominating the result.
- **generate**: stuff prompt with top-8 ranked node descriptions + top-`chunk_top_k`
  chunk text. Node prompt lines capped at 8 so chunk text gets enough attention.

## Storage schema

### PGVector — `rag_chunks` (one row per text chunk)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE rag_chunks (
    id          TEXT PRIMARY KEY,           -- sha256(source_path || ':' || chunk_index)
    source      TEXT NOT NULL,
    chunk_index INT  NOT NULL,
    text        TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   VECTOR({dim}) NOT NULL,     -- {dim} = embedding_dim setting (384 local, 1536 openai)
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX rag_chunks_source_idx ON rag_chunks (source);
```

### PGVector — `noderag_nodes` (one row per heterogeneous node)

```sql
CREATE TABLE noderag_nodes (
    id          TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL,              -- one of the six NodeRAG types
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   VECTOR({dim}) NOT NULL,
    chunk_ids   TEXT[] DEFAULT '{}',        -- back-references to rag_chunks
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX noderag_nodes_embedding_idx
    ON noderag_nodes USING hnsw (embedding vector_cosine_ops);
CREATE INDEX noderag_nodes_type_idx ON noderag_nodes (node_type);
```

### sqlite-graph — heterogeneous graph

```sql
-- managed by the sqlite-graph extension:
CREATE VIRTUAL TABLE graph USING graph();
-- exposes:
--   graph_nodes(id INTEGER, properties JSON, labels TEXT)
--   graph_edges(id INTEGER, source INTEGER, target INTEGER, edge_type TEXT, properties JSON)

-- plus our own name index since the extension uses integer node IDs:
CREATE TABLE entity_name_idx (
    name         TEXT PRIMARY KEY,          -- lowercased canonical name
    node_id      INTEGER NOT NULL UNIQUE,
    display_name TEXT
);
```

Entity property JSON shape: `{name, display_name, type, description, source_chunks: [chunk_id, ...]}`.
Relation property JSON shape: `{description, source_chunks: [chunk_id, ...], weight}`.

The extension is loaded into a regular `sqlite3` connection via
`conn.load_extension("vendor/libgraph.dylib")` — no daemon, no infra. Writes
go through `graph_node_add` / `graph_edge_add` SQL functions; reads use
direct `SELECT` over `graph_nodes` / `graph_edges`. Multi-hop expansion is
done in Python because v0.1.0-alpha Cypher has no variable-length paths.

## Ingestion

- `RecursiveCharacterTextSplitter`, `chunk_size=800`, `overlap=120`.
- Chunk ID is stable: `sha256(source_path + ":" + chunk_index)`. Re-ingest is a no-op.
- Embed in batches of 64 with the configured embedding provider (local default).
- PDF text is stripped of NUL / C0 control bytes after `pypdf.extract_text()` (Postgres rejects them in TEXT columns).
- `build_kg.py` per chunk: one DeepSeek extraction call returning
  `{entities: [{name, type, description}], relations: [{src, dst, type, description}]}`,
  then merged into sqlite-graph (lowercased name as merge key; descriptions
  concatenated; chunk_id appended to `source_chunks`).
- `build_noderag.py` (5-stage pipeline):
  1. Load chunks; per chunk LLM-extracts `semantic_units` (atomic claims) and `attributes` (entity.name = value)
  2. Harvest entities + relationships from sqlite-graph
  3. Embed and upsert all four base node types into `noderag_nodes`
  4. Build heterogeneous networkx graph (entity-entity from relations, semantic_unit↔entity, attribute↔entity), run Louvain community detection
  5. For each community: write a `high_level_element` membership node + DeepSeek-summarized `high_level_overview` node
- Both build scripts support `--limit N` for cost control.

## Interfaces

### CLI (`app/cli.py`)

```bash
python -m app.cli ask noderag "question" [--trace]
python -m app.cli ask {naive|agentic|graph|loop} "question"
python -m app.cli compare "question" [--patterns naive,agentic,graph,loop,noderag]
python -m app.cli ingest
python -m app.cli build-kg [--limit N]
python -m app.cli build-noderag [--limit N] [--skip-communities]
python -m app.cli noderag-stats
python -m app.cli reset --confirm
```

### FastAPI (`app/api.py`)

```
GET  /health                          → {"status": "ok"}
GET  /patterns                        → ["naive","agentic","graph","loop","noderag"]
POST /rag/{pattern}                   → {"question": "..."} → {"answer", "trace", "latency_ms"}
POST /ingest                          → {"chunks_added": N}
POST /kg/build       (?limit=N)       → {"entities": N, "relations": M}
POST /noderag/build  (?limit=N&skip_communities=BOOL) → counts per node type
GET  /noderag/stats                   → {"total": N, "by_type": {...}}
```

CORS open to `http://localhost:8501`.

### Streamlit (`app/ui.py`)

- Sidebar: pattern dropdown (auto-fetched from `/patterns`), ingest button, build-kg button, "show trace" toggle.
- Main panel: chat input + answer.
- Right panel (when trace on): rendered LangGraph history.
  - Naive: retrieved chunk IDs.
  - Agentic: tool-call timeline per agent turn.
  - Graph: matched entities + subgraph node/edge counts + chunks.
  - Loop: iteration table with align/faith/constraint deviation per row.
  - NodeRAG: seed-type breakdown, PPR expansion size, structural/semantic candidate counts.

The UI calls FastAPI; CLI imports the registry directly. API is canonical.

## Config

`.env` keys (see `.env.example`):

- `LLM_API_KEY`, `LLM_BASE_URL=https://api.deepseek.com/v1`, `LLM_MODEL=deepseek-chat`
- `EMBEDDING_PROVIDER=local|openai` (default `local`)
- `EMBEDDING_MODEL_LOCAL=BAAI/bge-small-en-v1.5`, `EMBEDDING_DIM=384`
- `EMBEDDING_MODEL_OPENAI=text-embedding-3-small` (set `EMBEDDING_DIM=1536` if switching)
- `OPENAI_API_KEY` (only required when `EMBEDDING_PROVIDER=openai`)
- `PG_DSN=postgresql://postgres:CHANGE_ME@localhost:5432/rag`
- `GRAPH_DB_PATH=data/graph.db`, `GRAPH_EXTENSION_PATH=vendor/libgraph.dylib`
- `LOOP_MAX_ITERATIONS=4`, `LOOP_CONVERGENCE_THRESHOLD=0.15`
- `NODERAG_TOP_K_SEEDS=20`, `NODERAG_PPR_TOP_N=50`, `NODERAG_PPR_ALPHA=0.15`, `NODERAG_CHUNK_TOP_K=18`
- `API_HOST=0.0.0.0`, `API_PORT=8000`, `STREAMLIT_API_URL=http://localhost:8000`
- `TOKENIZERS_PARALLELISM=false` (avoids HF fork warnings)

## Dependencies

Python 3.11+ (tested on 3.12). Single `pyproject.toml`:

- `langgraph`, `langchain`, `langchain-core`, `langchain-openai`, `langchain-community`
- `openai` (used for both OpenAI embeddings and the DeepSeek chat client via base-URL override)
- `sentence-transformers>=3.0,<4.0` (pin keeps `transformers<5.0` for torch 2.2 compat)
- `numpy<2.0` (torch 2.2 incompatible with numpy 2.x)
- `psycopg[binary]`, `pgvector`
- `networkx>=3.3`, `python-louvain>=0.16` (NodeRAG retrieval + community detection)
- `pypdf`, `tiktoken`
- `pydantic`, `pydantic-settings`, `python-dotenv`
- `typer`, `fastapi`, `uvicorn[standard]`, `streamlit`, `httpx`
- dev: `pytest`, `pytest-asyncio`, `ruff`, `ipykernel`, `jupyterlab`

**Removed:** `neo4j` (replaced by SQLite + sqlite-graph extension).

## First-run order

```bash
# Postgres (assumed already installed via brew or otherwise)
psql postgres -c "CREATE DATABASE rag;"
psql rag      -c "CREATE EXTENSION vector;"

# sqlite-graph C extension: prebuilt at vendor/libgraph.dylib (macOS x86_64).
# To rebuild for a different platform:
#   git clone https://github.com/agentflare-ai/sqlite-graph /tmp/sqlite-graph
#   cd /tmp/sqlite-graph && bash scripts/vendor.sh && make
#   cp build/libgraph.so /path/to/deepRAG/vendor/   # or .dylib

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env             # fill in LLM_API_KEY and PG_DSN password

python -m app.cli ingest         # ~30s on seed data
python -m app.cli build-kg       # ~2min on seed data, costs DeepSeek tokens
python -m app.cli build-noderag  # ~10-15min on a 7-page paper, costs more tokens

python -m app.cli ask naive "What is LangGraph?"
python -m app.cli compare "Who created LangGraph?"
uvicorn app.api:app --reload &
streamlit run app/ui.py
```

## Open questions / decisions made

- Hybrid vector + BM25 retriever? **Deferred** — NodeRAG's hybrid pool (PPR + cosine) covers this need for now.
- Agentic web search tool? **No** — keep local. Add only if a real use case appears.
- Loop RAG `check` learned scorer vs rule-based? **Rule-based** — matches the Bai et al. reference; LLM-graded faith axis already provides signal.
- Neo4j vs SQLite for the graph store? **SQLite + sqlite-graph extension** — zero infrastructure, vendored binary, file-based persistence. Multi-hop done in Python since v0.1.0-alpha Cypher lacks variable-length paths.
- OpenAI vs local embeddings? **Local default** (`BAAI/bge-small-en-v1.5`, 384-dim) — keeps document text on-machine, free, fast. OpenAI selectable via `EMBEDDING_PROVIDER=openai`.

## Success criteria

1. All five patterns answer the same question against the same corpus, producing
   visibly different traces — verified end-to-end on the LoopRAG paper.
2. Ingest, build-kg, and build-noderag are idempotent (re-running doesn't duplicate).
3. CLI, FastAPI, and Streamlit all reach all five patterns.
4. Loop RAG shows multiple iterations in its trace for ambiguous questions and
   converges in one iteration for trivial questions.
5. NodeRAG's PPR walk over the heterogeneous graph surfaces chunks the naive top-K
   would miss, and the hybrid candidate pool prevents structural bias from
   hiding semantically-relevant chunks.

## Status

All success criteria met as of 2026-05-21. Repo pushed to
https://github.com/leevydanomalik/deepRAG. Validated against the LoopRAG
paper (Bai et al. 2026): 161 chunks indexed in pgvector, 603 entities + 700
relations in sqlite-graph, 3,075 nodes (across 6 types) in `noderag_nodes`,
with 75 Louvain communities and 69 LLM-summarized overviews.
