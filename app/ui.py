"""Streamlit UI: pick a pattern, ask a question, inspect the trace.
Calls the FastAPI server (so the API is the canonical interface)."""
from __future__ import annotations

import os

import httpx
import streamlit as st

API = os.environ.get("STREAMLIT_API_URL", "http://localhost:8000")

# ---------- preset questions (used by sidebar buttons) ----------

SAMPLE_QUESTIONS = [
    ("What is LangGraph?", "naive"),
    ("Who created LangChain and when?", "naive"),
    ("What does the Plan agent do in LoopRAG?", "naive"),
    ("Who authored the LoopRAG paper and what concepts do they introduce?", "graph"),
]

COMPLEX_QUESTIONS = [
    (
        "What is LoopRAG's three-axis deviation signal, and what does each axis measure?",
        "noderag",
    ),
    (
        "Compare the Plan agent and the Act agent in LoopRAG — what does each do "
        "and how do they relate?",
        "agentic",
    ),
    (
        "What ecosystem of products has LangChain Inc. built, what does each do, "
        "and which one would I use for a PDCA-style closed-loop RAG?",
        "graph",
    ),
    (
        "What is the overall accuracy improvement LoopRAG achieves compared to the "
        "baseline RAG system, and which components contribute most?",
        "loop",
    ),
    (
        "Identify three concepts or techniques mentioned in the LoopRAG paper that "
        "LoopRAG does NOT use or depend on. For each, explain why it appears in the "
        "paper.",
        "agentic",
    ),
]

st.set_page_config(page_title="deepRAG", layout="wide")
st.title("deepRAG — Five RAG Patterns")
st.caption(
    "naive · agentic · graph · loop (PDCA) · noderag (HNSW + PPR)"
)


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


with st.sidebar:
    st.header("Settings")
    try:
        patterns = httpx.get(f"{API}/patterns", timeout=5).json()
    except Exception:
        patterns = ["naive", "agentic", "graph", "loop", "noderag"]
        st.warning(f"API unreachable at {API}; using static pattern list.")

    # Pattern selector — driven by session_state so sample buttons can override
    pattern = st.selectbox(
        "Pattern",
        patterns,
        index=patterns.index(st.session_state.get("pattern", patterns[0]))
        if st.session_state.get("pattern") in patterns
        else 0,
        key="pattern",
    )
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
    if st.button("Build NodeRAG"):
        with st.spinner("Building NodeRAG index (~10 min)…"):
            r = httpx.post(f"{API}/noderag/build", timeout=3600)
            st.success(r.json())

    st.markdown("---")
    st.subheader("Sample questions")
    st.caption("Click to prefill — direct fact / single-chunk recall")
    for i, (q, pat) in enumerate(SAMPLE_QUESTIONS):
        if st.button(_short(q), key=f"sample_{i}", use_container_width=True):
            st.session_state["question"] = q
            st.session_state["pattern"] = pat
            st.rerun()

    st.markdown("---")
    st.subheader("Complex / multi-hop")
    st.caption("Compare patterns on hard questions")
    for i, (q, pat) in enumerate(COMPLEX_QUESTIONS):
        if st.button(_short(q), key=f"complex_{i}", use_container_width=True):
            st.session_state["question"] = q
            st.session_state["pattern"] = pat
            st.rerun()

    st.markdown("---")
    st.caption(
        "Sample = 1-shot retrieval is enough.  \n"
        "Complex = needs multi-step / multi-hop / hybrid retrieval."
    )


col_a, col_b = st.columns([2, 1] if show_trace else [1, 0.001])

with col_a:
    question = st.text_area(
        "Question",
        height=120,
        placeholder="Ask anything… or click a preset in the sidebar.",
        key="question",
    )
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
                st.caption(f"pattern: {pattern}  ·  latency: {data['latency_ms']} ms")
                st.session_state["last_trace"] = data["trace"]

with col_b:
    if show_trace and st.session_state.get("last_trace"):
        st.markdown("### Trace")
        st.json(st.session_state["last_trace"])
