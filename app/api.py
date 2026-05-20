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


@app.post("/noderag/build")
def noderag_build(limit: int | None = None, skip_communities: bool = False):
    from scripts.build_noderag import run_build_noderag
    return run_build_noderag(limit=limit, skip_communities=skip_communities)


@app.get("/noderag/stats")
def noderag_stats():
    from rag.core.noderag_store import NodeRAGStore
    nr = NodeRAGStore()
    out = {"total": nr.count(), "by_type": nr.counts_by_type()}
    nr.close()
    return out
