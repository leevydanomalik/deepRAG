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
