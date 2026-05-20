"""Streamlit UI: pick a pattern, ask a question, inspect the trace.
Calls the FastAPI server (so the API is the canonical interface)."""
from __future__ import annotations

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
