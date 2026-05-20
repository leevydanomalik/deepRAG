# deepRAG — Architecture & Dataflow Diagrams

All diagrams are Mermaid; GitHub renders them natively when you open this file.

## Contents

1. [System overview](#1-system-overview)
2. [Offline pipelines](#2-offline-pipelines) — `ingest`, `build-kg`, `build-noderag`
3. [Storage schema](#3-storage-schema)
4. [Pattern dataflows](#4-pattern-dataflows)
   - [Naive RAG](#41-naive-rag)
   - [Agentic RAG](#42-agentic-rag)
   - [Graph RAG](#43-graph-rag)
   - [Loop RAG (PDCA)](#44-loop-rag-pdca)
   - [NodeRAG](#45-noderag)
5. [Request/response sequence](#5-requestresponse-sequence)

---

## 1. System overview

```mermaid
flowchart LR
  subgraph CL["Clients"]
    UI["Streamlit UI<br/>localhost:8501"]
    CLI["Typer CLI<br/>python -m app.cli"]
    NB["Jupyter notebooks"]
  end

  subgraph APP["App layer (app/)"]
    API["FastAPI<br/>localhost:8000"]
    REG["registry.py<br/>5 build_graph fns"]
  end

  subgraph PATTERNS["RAG patterns (rag/)"]
    P1["naive"]
    P2["agentic"]
    P3["graph_rag"]
    P4["loop"]
    P5["noderag"]
  end

  subgraph CORE["Core (rag/core/)"]
    LLM["llm.py<br/>DeepSeek chat<br/>OpenAI-compatible"]
    EMB["embeddings.py<br/>BAAI/bge-small-en-v1.5<br/>(local 384-dim)"]
    PG["pg_store.py"]
    GS["graph_store.py<br/>sqlite-graph"]
    NRS["noderag_store.py"]
    LD["loader.py<br/>PDF/MD/TXT"]
    PROMPT["prompts.py"]
  end

  subgraph STORES["Storage (local)"]
    PGSQL[("PostgreSQL + pgvector<br/>rag DB")]
    SQL[("SQLite + sqlite-graph<br/>data/graph.db")]
    DEEP{{"DeepSeek API<br/>api.deepseek.com/v1"}}
  end

  UI -->|"HTTP"| API
  CLI -->|"in-process"| REG
  NB -->|"in-process"| REG
  API -->|"in-process"| REG

  REG --> P1 & P2 & P3 & P4 & P5

  P1 --> EMB & PG & LLM
  P2 --> EMB & PG & GS & LLM
  P3 --> EMB & PG & GS & LLM
  P4 --> EMB & PG & GS & LLM
  P5 --> EMB & NRS & GS & LLM

  PG --> PGSQL
  NRS --> PGSQL
  GS --> SQL
  LLM --> DEEP
  EMB -. "model file<br/>~80 MB" .-> EMB

  LD -.-> P1 & P2 & P3 & P4 & P5
  PROMPT -.-> P1 & P2 & P3 & P4 & P5

  classDef store fill:#e1f5ff,stroke:#0369a1
  classDef pattern fill:#fef3c7,stroke:#b45309
  classDef ext fill:#fae8ff,stroke:#86198f
  class PGSQL,SQL store
  class P1,P2,P3,P4,P5 pattern
  class DEEP ext
```

---

## 2. Offline pipelines

Run once after dropping documents into `data/raw/`. All three are idempotent.

### 2.1. `ingest` — chunk → embed → upsert

```mermaid
flowchart LR
  RAW["data/seed/*<br/>data/raw/*"] --> LOAD["loader.load_documents()<br/>PDF / MD / TXT"]
  LOAD --> STRIP["_strip_control_chars()<br/>(NUL byte removal)"]
  STRIP --> CHUNK["chunk_document()<br/>RecursiveCharacterTextSplitter<br/>size=800, overlap=120"]
  CHUNK --> ID["stable id =<br/>sha256(path:chunk_index)"]
  ID --> BATCH["batch of 64"]
  BATCH --> EMB["embed_batch()<br/>BAAI/bge-small-en-v1.5"]
  EMB --> UPSERT["PgStore.upsert()<br/>ON CONFLICT (id)<br/>DO UPDATE"]
  UPSERT --> PG[("rag_chunks<br/>(pgvector)")]

  classDef store fill:#e1f5ff,stroke:#0369a1
  class PG store
```

### 2.2. `build-kg` — KG extraction → sqlite-graph

```mermaid
flowchart LR
  PG[("rag_chunks")] --> FETCH["SELECT id, text<br/>FROM rag_chunks"]
  FETCH --> LOOP{"per chunk"}
  LOOP --> EX["LLM extract:<br/>KG_EXTRACTION_PROMPT"]
  EX --> PARSE["entities[]<br/>relations[]"]
  PARSE --> MERGE_E["GraphStore.merge_entity()<br/>name (lowercased) is merge key<br/>descriptions concatenated<br/>chunk_ids appended"]
  PARSE --> MERGE_R["GraphStore.merge_relation()<br/>(src, dst, type) is merge key<br/>weight incremented<br/>chunk_ids appended"]
  MERGE_E --> SQL[("graph_nodes<br/>+ entity_name_idx<br/>(sqlite-graph)")]
  MERGE_R --> SQL

  classDef store fill:#e1f5ff,stroke:#0369a1
  class PG,SQL store
```

### 2.3. `build-noderag` — 5-stage heterogeneous-node pipeline

```mermaid
flowchart TB
  S1["Stage 1<br/>load chunks from rag_chunks"]
  S2["Stage 2 — per-chunk LLM extractions<br/>NODERAG_SEMANTIC_UNIT_PROMPT → atomic claims<br/>NODERAG_ATTRIBUTE_PROMPT → entity.name = value"]
  S3["Stage 3<br/>harvest entities + relationships<br/>from sqlite-graph"]
  S4["Stage 4 — embed + upsert<br/>entity, relationship, semantic_unit, attribute<br/>into noderag_nodes"]
  S5_A["Stage 5a — networkx graph<br/>entity-entity (1.0)<br/>semantic_unit↔entity (0.7)<br/>attribute↔entity (0.5)"]
  S5_B["Stage 5b — Louvain communities<br/>(python-louvain)"]
  S5_C["Stage 5c — per community ≥3 members<br/>high_level_element (membership)<br/>high_level_overview (LLM summary)"]

  S1 --> S2 --> S3 --> S4 --> S5_A --> S5_B --> S5_C
  S5_C --> NR[("noderag_nodes<br/>6 types, pgvector<br/>+ HNSW index")]

  classDef store fill:#e1f5ff,stroke:#0369a1
  class NR store
```

---

## 3. Storage schema

```mermaid
erDiagram
    RAG_CHUNKS {
        text id PK "sha256(path:idx)"
        text source
        int  chunk_index
        text text
        jsonb metadata
        vector embedding "384 or 1536 dim"
        timestamptz created_at
    }

    NODERAG_NODES {
        text id PK "ent|rel|unit|attr|hle|hlo : hash"
        text node_type "6 allowed values"
        text content
        jsonb metadata
        vector embedding
        text_array chunk_ids "back-ref to rag_chunks"
        timestamptz created_at
    }

    GRAPH_NODES {
        int id PK "integer node id"
        text properties "JSON: name, display_name, type, description, source_chunks"
        text labels
    }

    GRAPH_EDGES {
        int id PK
        int source FK
        int target FK
        text edge_type
        text properties "JSON: description, source_chunks, weight"
    }

    ENTITY_NAME_IDX {
        text name PK "lowercased canonical"
        int  node_id FK "to graph_nodes.id"
        text display_name
    }

    RAG_CHUNKS ||--o{ NODERAG_NODES : "chunk_ids[] (text array)"
    RAG_CHUNKS ||--o{ GRAPH_NODES   : "properties.source_chunks[]"
    RAG_CHUNKS ||--o{ GRAPH_EDGES   : "properties.source_chunks[]"
    GRAPH_NODES }o--|| ENTITY_NAME_IDX : "node_id"
    GRAPH_NODES ||--o{ GRAPH_EDGES : "source / target"
```

Tables `rag_chunks` and `noderag_nodes` live in **PostgreSQL** (pgvector). Tables
`graph_nodes`, `graph_edges`, `entity_name_idx` live in **SQLite** at
`data/graph.db` (sqlite-graph extension). Cross-store references are by
chunk ID string — no FK enforcement across stores.

---

## 4. Pattern dataflows

All five patterns expose the same interface:

```python
def build_graph() -> CompiledStateGraph: ...
# input:  {"question": str}
# output: {"answer": str, "history": list[dict], ...}
```

### 4.1. Naive RAG

```mermaid
flowchart LR
  Q[/"question"/] --> R["retrieve<br/>embed_text(q)<br/>top-5 cosine"]
  R --> CTX["context = top-5 chunks<br/>(stuffed)"]
  CTX --> G["generate<br/>DeepSeek + ANSWER_PROMPT"]
  G --> A[/"answer"/]

  R -.-> PG[("rag_chunks")]
  classDef store fill:#e1f5ff,stroke:#0369a1
  class PG store
```

### 4.2. Agentic RAG

```mermaid
flowchart LR
  Q[/"question"/] --> AG["agent<br/>DeepSeek + SYSTEM_PROMPT<br/>bind_tools([vector_search, kg_lookup])"]
  AG -->|"tool_calls?"| COND{tools_condition}
  COND -->|"yes"| T["ToolNode<br/>parallel tool execution"]
  COND -->|"no"| END[/"final answer"/]
  T --> AG

  T -.->|"vector_search"| PG[("rag_chunks")]
  T -.->|"kg_lookup"| SQL[("sqlite-graph")]

  classDef store fill:#e1f5ff,stroke:#0369a1
  class PG,SQL store
```

Agent decides when to stop (no more tool calls → emit final answer). Recursion
limit caps runaway loops.

### 4.3. Graph RAG

```mermaid
flowchart LR
  Q[/"question"/] --> EE["extract_query_entities<br/>DeepSeek +<br/>GRAPH_RAG_ENTITY_EXTRACT_PROMPT"]
  EE --> ES["expand_subgraph<br/>2-hop BFS in Python<br/>over sqlite-graph"]
  ES --> FC["fetch_chunks<br/>union of source_chunks<br/>from matched nodes/edges"]
  FC --> G["generate<br/>DeepSeek + ANSWER_PROMPT<br/>subgraph summary + chunks"]
  G --> A[/"answer"/]

  ES -.-> SQL[("sqlite-graph")]
  FC -.-> PG[("rag_chunks")]

  classDef store fill:#e1f5ff,stroke:#0369a1
  class PG,SQL store
```

### 4.4. Loop RAG (PDCA)

```mermaid
flowchart LR
  Q[/"question"/] --> P["plan<br/>classify task_type<br/>extract entities/constraints<br/>pick prompt template"]
  P --> D["do<br/>hybrid retrieve<br/>(pgvector + sqlite-graph)<br/>generate answer"]
  D --> C["check<br/>align = 1 - cos(E(a),E(q))<br/>faith = LLM-graded support<br/>constraint = length/structure heuristic<br/>total = 0.5·a + 0.4·f + 0.1·c"]
  C --> COND{"total < threshold<br/>OR iter ≥ max?"}
  COND -->|"yes"| END[/"converged answer"/]
  COND -->|"no"| ACT["act<br/>rewrite query based on<br/>dominant deviation axis<br/>swap prompt template"]
  ACT --> P

  D -.-> PG[("rag_chunks")]
  D -.-> SQL[("sqlite-graph")]

  classDef store fill:#e1f5ff,stroke:#0369a1
  class PG,SQL store
```

Recursion budget: `max_iterations × 4 + 4` graph steps. Each PDCA cycle is one
iteration; convergence threshold from `LOOP_CONVERGENCE_THRESHOLD` env var.

### 4.5. NodeRAG

```mermaid
flowchart TB
  Q[/"question"/] --> EMBED["embed_query<br/>cache q_embedding on state"]
  EMBED --> VS["vector_seed<br/>HNSW top-K across<br/>all 6 node types"]
  VS --> PPR["ppr_propagate<br/>networkx.pagerank()<br/>α = 1 - ppr_alpha<br/>personalization =<br/>cosine_score × type_weight"]
  PPR --> FC["fetch_chunks<br/>HYBRID candidate pool"]
  FC --> RR["cosine re-rank<br/>over merged pool"]
  RR --> G["generate<br/>top-8 ranked nodes +<br/>top-K chunks"]
  G --> A[/"answer"/]

  subgraph HYB["fetch_chunks internals"]
    direction LR
    HA["(a) round-robin chunks<br/>from PPR-ranked nodes<br/>STRUCTURAL relevance"]
    HB["(b) top-K cosine on chunks<br/>SEMANTIC relevance"]
    HMERGE["merge ∪ dedupe"]
    HA --> HMERGE
    HB --> HMERGE
  end
  FC -.-> HYB
  HYB -.-> RR

  VS -.-> NR[("noderag_nodes")]
  PPR -.-> GRAPH["heterogeneous<br/>networkx graph<br/>(LRU-cached)"]
  HB -.-> PG[("rag_chunks")]
  HA -.-> NR

  classDef store fill:#e1f5ff,stroke:#0369a1
  classDef cache fill:#fef3c7,stroke:#b45309
  class PG,NR store
  class GRAPH cache
```

**Seed-type weights** applied to PPR personalization:

| Type | Weight | Rationale |
|---|---:|---|
| semantic_unit | 1.4 | atomic claims carry full factual text |
| high_level_overview | 1.3 | LLM-written summaries |
| high_level_element | 1.1 | community labels |
| entity | 1.0 | baseline |
| attribute | 1.0 | baseline |
| relationship | 0.6 | rarely carries the underlying factual text |

The hybrid pool is the key fix: PPR alone walks toward chunks tied to
highly-connected entities and **misses detail-heavy chunks** (formulas, tables)
that have few graph neighbors. Direct cosine restores those; final rerank
orders by query relevance regardless of which path surfaced each chunk.

---

## 5. Request/response sequence

End-to-end on an "ask" call hitting the FastAPI server, for **NodeRAG** as an
example (other patterns differ only in the pattern-internal steps).

```mermaid
sequenceDiagram
  participant U as User (UI / CLI)
  participant API as FastAPI
  participant REG as registry.py
  participant NR as rag.noderag.graph
  participant EMB as embeddings.py
  participant PG as PgStore (pgvector)
  participant NRS as NodeRAGStore
  participant SQL as GraphStore (sqlite)
  participant NX as networkx
  participant LLM as DeepSeek

  U->>API: POST /rag/noderag {question}
  API->>REG: run("noderag", question)
  REG->>NR: app.invoke({question})

  NR->>EMB: embed_text(q)
  EMB-->>NR: q_embedding (384-dim)

  NR->>NRS: vector_search(q_embedding, k=20)
  NRS->>PG: SELECT ... ORDER BY embedding <=> q LIMIT 20
  PG-->>NRS: 20 seed nodes
  NRS-->>NR: seed_nodes

  Note over NR,NX: PPR step
  NR->>NRS: all_nodes_minimal()
  NR->>SQL: relations + node-type membership
  NR->>NX: pagerank(g, personalization=seeds)
  NX-->>NR: top-50 ranked node ids
  NR->>NRS: fetch_by_ids(top-50)

  Note over NR,PG: hybrid candidate pool
  NR->>PG: similarity_search (top-K direct cosine)
  PG-->>NR: semantic candidates
  NR->>PG: ORDER BY <=> q over merged pool LIMIT 18
  PG-->>NR: 18 reranked chunks

  NR->>LLM: ANSWER_PROMPT(context, question)
  LLM-->>NR: answer text

  NR-->>REG: {answer, history, ranked_nodes, ...}
  REG-->>API: {answer, trace, raw}
  API-->>U: 200 OK {answer, trace, latency_ms}
```

The other four patterns follow the same client → API → registry → pattern
shape but differ in the middle:

- **naive**: one `retrieve` round-trip to `PgStore`, one `LLM` call.
- **agentic**: N round-trips of `LLM` ⇄ tools (`PgStore.similarity_search` or `GraphStore.get_entity`), driven by LLM tool-calling.
- **graph**: one LLM extraction, one BFS in Python over `GraphStore`, one chunk fetch on `PgStore`, one LLM generation.
- **loop**: 1-4 PDCA iterations; each iteration is `plan (LLM) → do (PgStore + GraphStore + LLM) → check (embeddings + LLM)` and optionally `act (LLM)`.
