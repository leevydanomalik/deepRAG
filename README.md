# DIP — Four RAG Patterns in LangGraph

Naive, Agentic, Graph, and Loop (PDCA) RAG implementations sharing one
**PostgreSQL+pgvector** store and one **SQLite + [sqlite-graph](https://github.com/agentflare-ai/sqlite-graph)**
knowledge graph, exposed via CLI, FastAPI, and Streamlit.

## Quick start

```bash
# 1. Postgres + pgvector (assumed already installed via brew)
psql postgres -c "CREATE DATABASE rag;"
psql rag      -c "CREATE EXTENSION vector;"

# 2. Build the sqlite-graph extension (already done — vendored at vendor/libgraph.dylib for macOS)
#    If you need to rebuild from source:
#      git clone https://github.com/agentflare-ai/sqlite-graph /tmp/sqlite-graph
#      cd /tmp/sqlite-graph && bash scripts/vendor.sh && make
#      cp build/libgraph.dylib /path/to/DIP/vendor/

# 3. Python env
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then edit: LLM_API_KEY, PG_DSN, GRAPH_EXTENSION_PATH

# 4. Index your corpus + extract KG
python -m app.cli ingest
python -m app.cli build-kg

# 5. Ask
python -m app.cli ask naive "What is LangGraph?"

# 6. Or serve
uvicorn app.api:app --reload &
streamlit run app/ui.py
```

## Patterns

| Pattern  | Topology                                                          | When to use |
|----------|-------------------------------------------------------------------|-------------|
| naive    | retrieve → generate                                               | baseline |
| agentic  | agent ⇄ tools (vector_search, kg_lookup)                          | multi-step lookup |
| graph    | extract entities → expand subgraph (2-hop BFS) → fetch chunks → generate | relationship queries |
| loop     | plan → do → check ⇄ act (PDCA, three-axis deviation)              | hard/ambiguous queries |

## Stack

| Concern         | Choice |
|-----------------|--------|
| LLM (chat)      | DeepSeek `deepseek-chat` via OpenAI-compatible API |
| Embeddings      | Local `BAAI/bge-small-en-v1.5` via sentence-transformers (384-dim) |
| Vector store    | PostgreSQL + pgvector |
| Graph store     | SQLite + [sqlite-graph](https://github.com/agentflare-ai/sqlite-graph) extension |
| Orchestration   | LangGraph 0.2 |
| CLI / API / UI  | Typer / FastAPI / Streamlit |

The graph store wraps sqlite-graph's `graph_node_add` / `graph_edge_add` SQL
functions and reads via direct SQL on the `graph_nodes` / `graph_edges`
virtual table. Multi-hop expansion is done in Python (BFS) since the
v0.1.0-alpha Cypher doesn't yet support variable-length paths.

## Run tests

```bash
pytest -m "not integration and not llm"   # fast, no PG/network needed
pytest -m integration                      # needs PG (Neo4j is gone)
pytest -m llm                              # costs DeepSeek tokens
```

## Layout

```
rag/core/                            LLM, embeddings, pg_store, graph_store, loader, prompts
rag/{naive,agentic,graph_rag,loop}/  one LangGraph per pattern
rag/registry.py                      pattern name → graph builder
scripts/                             ingest, build_kg, reset_stores
app/                                 cli, api, ui
vendor/libgraph.dylib                sqlite-graph C extension, built for macOS x86_64
data/seed/                           bundled sample corpus
data/raw/                            drop your own PDFs/MD/TXT here
data/graph.db                        SQLite graph DB (gitignored)
notebooks/                           per-pattern Jupyter notebooks
```

See `docs/superpowers/specs/2026-05-20-rag-patterns-langgraph-design.md` for
the design and `docs/superpowers/plans/2026-05-20-rag-patterns-langgraph.md`
for the implementation plan.
