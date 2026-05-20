# DIP — Four RAG Patterns in LangGraph (Design)

**Status:** Draft for review
**Date:** 2026-05-20
**Owner:** Leevyd Malik

## Goal

Build a single Python project that implements **four RAG patterns as LangGraph
graphs** sharing one ingestion pipeline, one vector store (PGVector), and one
knowledge graph (Neo4j). Expose all four through a CLI, a FastAPI server, and a
Streamlit UI so they can be compared interactively.

The four patterns:

1. **Naive RAG** — retrieve once, generate.
2. **Agentic RAG** — LLM tool-calls a retriever until it has enough context.
3. **Graph RAG** — entity extraction + Neo4j subgraph traversal + chunk lookup.
4. **Loop RAG** — PDCA (Plan-Do-Check-Act) closed-loop multi-agent, iterating
   until a three-axis deviation signal converges. Mirrors the pattern in
   `/Users/leevydmalik/Documents/Project/DAYA/LOOPRAG`.

## Non-goals

- Production deployment (auth, rate limiting, multi-tenant) — single user, local.
- Evaluation harness (RAGAS, etc.) — out of scope for this milestone.
- Custom embedding models — `text-embedding-3-small` is enough.
- Streaming responses — first cut is request/response only.

## Stack

| Concern         | Choice                                        |
|-----------------|-----------------------------------------------|
| LLM (chat)      | DeepSeek `deepseek-chat` via OpenAI-compatible API |
| Embeddings      | OpenAI `text-embedding-3-small` (1536-dim)    |
| Vector store    | PostgreSQL 16 + `pgvector` (local, brew)      |
| Graph store     | Neo4j 5 (local, brew)                         |
| Orchestration   | LangGraph 0.2                                 |
| CLI             | Typer                                         |
| API             | FastAPI + uvicorn                             |
| UI              | Streamlit                                     |
| Python          | 3.11+, managed with `uv`                      |

## Repository layout

```
DIP/
├── pyproject.toml
├── .env.example
├── README.md
├── data/
│   ├── seed/                  # bundled sample markdown (LangGraph/LangChain docs)
│   └── raw/                   # user drops PDFs/MD/TXT here
├── scripts/
│   ├── ingest.py              # chunk → embed → upsert to pgvector
│   ├── build_kg.py            # LLM-extract entities/relations → Neo4j
│   └── reset_stores.py        # truncate pgvector + wipe neo4j (with --confirm)
├── rag/
│   ├── core/
│   │   ├── llm.py             # DeepSeek chat client
│   │   ├── embeddings.py      # OpenAI text-embedding-3-small
│   │   ├── pg_store.py        # pgvector retrieval (cosine + hybrid)
│   │   ├── neo4j_store.py     # graph traversal helpers
│   │   ├── loader.py          # PDF/MD/TXT loaders + chunking
│   │   └── config.py          # pydantic-settings, loads .env
│   ├── naive/
│   │   ├── state.py
│   │   └── graph.py
│   ├── agentic/
│   │   ├── state.py
│   │   ├── tools.py           # vector_search, kg_lookup
│   │   └── graph.py
│   ├── graph_rag/
│   │   ├── state.py
│   │   └── graph.py
│   ├── loop/
│   │   ├── state.py           # mirrors LoopRAGState (intent/taskGraph/...)
│   │   ├── agents.py          # plan_node, do_node, check_node, act_node
│   │   └── graph.py
│   └── registry.py            # pattern name → build_graph fn
├── app/
│   ├── cli.py                 # `python -m app.cli ask <pattern> "..."`
│   ├── api.py                 # FastAPI: POST /rag/{pattern}
│   └── ui.py                  # Streamlit dropdown + chat + trace panel
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
START → extract_query_entities → expand_subgraph → fetch_chunks → rerank → generate → END
```

- `extract_query_entities`: DeepSeek extracts entity mentions from the question.
- `expand_subgraph`: Cypher query — `MATCH (e:Entity)-[r*1..2]-(n) WHERE e.name IN $names`.
- `fetch_chunks`: collect `source_chunks` IDs from matched entities/relations, then fetch
  full text from `rag_chunks`.
- `rerank`: combine entity-name match score + chunk cosine similarity via reciprocal
  rank fusion; keep top 8.
- `generate`: stuff prompt with subgraph summary + chunks.

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
- **do**: hybrid retrieve (pgvector cosine + Neo4j subgraph from plan entities) →
  generate answer.
- **check**: compute three-axis deviation
  - `align = 1 − cos(E(answer), E(question))`
  - `faith = 1 − mean(δᵢ)` where δᵢ is per-evidence support score
  - `constraint = 1 − Ψ(answer, prompt_policy)` (lexical+structural rule check)
  - `total = w_a·align + w_f·faith + w_c·constraint`
  - converged when `total < LOOP_CONVERGENCE_THRESHOLD` or `iteration >= LOOP_MAX_ITERATIONS`.
- **act**: rewrite intent + swap template based on dominant deviation axis.

State is the same shape as the reference `LoopRAGState` (intent, task_graph,
prompt_config, evidence, answer, deviation, iteration, converged, history).

## Storage schema

### PGVector

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE rag_chunks (
    id          TEXT PRIMARY KEY,           -- sha256(source_path || ':' || chunk_index)
    source      TEXT NOT NULL,
    chunk_index INT  NOT NULL,
    text        TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   VECTOR(1536) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX rag_chunks_source_idx ON rag_chunks (source);
```

Optional hybrid: add `tsvector` column + GIN index, fuse with cosine via RRF.

### Neo4j

```cypher
(:Entity {id, name, type, description, source_chunks: [text...]})
(:Chunk  {id, source, text})

(:Entity)-[:RELATES {type, description, weight, source_chunks: [text...]}]->(:Entity)
(:Entity)-[:MENTIONED_IN]->(:Chunk)
```

`source_chunks` lets Graph RAG map matched entities/relations back to chunk text in pgvector.

## Ingestion

- `RecursiveCharacterTextSplitter`, `chunk_size=800`, `overlap=120`.
- Chunk ID is stable: `sha256(source_path + ":" + chunk_index)`. Re-ingest is a no-op.
- Embed in batches of 64 with `text-embedding-3-small`.
- `build_kg.py` per chunk: one DeepSeek extraction call returning
  `{entities: [{name, type, description}], relations: [{src, dst, type, description}]}`,
  then MERGE into Neo4j (lowercased name as merge key; descriptions concatenated;
  chunk_id appended to `source_chunks`).
- `build_kg.py` supports `--limit N` so large `data/raw/` doesn't burn the budget by accident.

## Interfaces

### CLI (`app/cli.py`)

```bash
python -m app.cli ask naive   "question"
python -m app.cli ask agentic "question"
python -m app.cli ask graph   "question"
python -m app.cli ask loop    "question"
python -m app.cli ingest
python -m app.cli build-kg [--limit N]
python -m app.cli reset --confirm
```

### FastAPI (`app/api.py`)

```
GET  /health                          → {"status": "ok"}
GET  /patterns                        → ["naive","agentic","graph","loop"]
POST /rag/{pattern}                   → {"question": "..."} → {"answer", "trace", "latency_ms"}
POST /ingest                          → {"chunks_added": N}
POST /kg/build  (?limit=N)            → {"entities": N, "relations": M}
```

CORS open to `http://localhost:8501`.

### Streamlit (`app/ui.py`)

- Sidebar: pattern dropdown, ingest button, build-kg button, "show trace" toggle.
- Main panel: chat input + answer.
- Right panel (when trace on): rendered LangGraph history.
  - Naive: retrieved chunks.
  - Agentic: tool-call timeline.
  - Graph: matched entities + subgraph edges + chunks.
  - Loop: iteration table with align/faith/constraint per row.

The UI calls FastAPI; CLI imports the registry directly. API is canonical.

## Config

`.env` keys (see `.env.example`):

- `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL=https://api.deepseek.com`, `DEEPSEEK_MODEL=deepseek-chat`
- `OPENAI_API_KEY`, `EMBEDDING_MODEL=text-embedding-3-small`, `EMBEDDING_DIM=1536`
- `PG_DSN=postgresql://postgres:postgres@localhost:5432/dip_rag`
- `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=...`
- `LOOP_MAX_ITERATIONS=4`, `LOOP_CONVERGENCE_THRESHOLD=0.15`
- `API_HOST=0.0.0.0`, `API_PORT=8000`, `STREAMLIT_API_URL=http://localhost:8000`

## Dependencies

Python 3.11+. Single `pyproject.toml`:

- langgraph, langchain, langchain-core, langchain-openai, langchain-community
- openai
- psycopg[binary], pgvector
- neo4j
- pypdf, tiktoken
- pydantic, pydantic-settings, python-dotenv
- typer, fastapi, uvicorn[standard], streamlit, httpx
- dev: pytest, pytest-asyncio, ruff, ipykernel, jupyterlab

## First-run order

```bash
brew install postgresql@16 && brew services start postgresql@16
brew install neo4j         && brew services start neo4j
psql postgres -c "CREATE DATABASE dip_rag;"
psql dip_rag  -c "CREATE EXTENSION vector;"

uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env             # fill in DEEPSEEK_API_KEY, OPENAI_API_KEY, NEO4J_PASSWORD

python -m app.cli ingest         # ~30s on seed data
python -m app.cli build-kg       # ~2min on seed data, costs DeepSeek tokens

python -m app.cli ask naive "What is LangGraph?"
uvicorn app.api:app --reload &
streamlit run app/ui.py
```

## Open questions

- Do we want a hybrid (vector + BM25) retriever as the default, or keep vector-only
  for the first cut? **Decision: vector-only first, add hybrid as a follow-on.**
- Should the Agentic tool set include a web search? **Decision: no — keep it local.
  Add later if needed.**
- Should Loop RAG's `check` node use a learned scorer or rule-based? **Decision:
  rule-based for first cut, matches the reference implementation.**

## Success criteria

1. All four patterns answer the same question against the same corpus, producing
   visibly different traces.
2. Ingest + build-kg are idempotent (re-running doesn't duplicate).
3. CLI, FastAPI, and Streamlit all reach all four patterns.
4. Loop RAG shows multiple iterations in its trace for ambiguous questions and
   converges in one iteration for trivial questions.
