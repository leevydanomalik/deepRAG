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

| Pattern  | Topology                                                  | When to use |
|----------|-----------------------------------------------------------|-------------|
| naive    | retrieve → generate                                       | baseline |
| agentic  | agent ⇄ tools (vector_search, kg_lookup)                  | multi-step lookup |
| graph    | extract entities → expand Neo4j → fetch chunks → generate | relationship queries |
| loop     | plan → do → check ⇄ act (PDCA)                            | hard/ambiguous queries |

## Run tests

```bash
pytest -m "not integration and not llm"   # fast, no infra needed
pytest -m integration                      # needs PG + Neo4j running
pytest -m llm                              # costs DeepSeek tokens
```

## Layout

```
rag/core/                            shared LLM, embeddings, stores, loader, prompts
rag/{naive,agentic,graph_rag,loop}/  one LangGraph per pattern
rag/registry.py                      pattern name → graph builder
scripts/                             ingest, build_kg, reset_stores
app/                                 cli, api, ui
data/seed/                           bundled sample corpus
data/raw/                            drop your own PDFs/MD/TXT here
notebooks/                           per-pattern Jupyter notebooks
```

See `docs/superpowers/specs/2026-05-20-rag-patterns-langgraph-design.md` for
the design and `docs/superpowers/plans/2026-05-20-rag-patterns-langgraph.md`
for the implementation plan.
