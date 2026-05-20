# RAG Patterns — A Theoretical Primer

A conceptual companion to the deepRAG project. Each section explains the
*problem* a pattern was invented to solve, the *key insight* behind its
design, and *when it shines vs. when it fails*. Implementations of all five
patterns live in [`rag/`](../rag/); architectural diagrams in
[`docs/diagrams/architecture.md`](diagrams/architecture.md).

## Contents

1. [What problem is RAG solving?](#1-what-problem-is-rag-solving)
2. [Naive RAG](#2-naive-rag)
3. [Agentic RAG](#3-agentic-rag)
4. [Graph RAG](#4-graph-rag)
5. [Loop RAG (PDCA)](#5-loop-rag-pdca)
6. [NodeRAG (heterogeneous nodes + PPR)](#6-noderag-heterogeneous-nodes--ppr)
7. [Comparison matrix](#7-comparison-matrix)
8. [Decision guide — which to pick](#8-decision-guide--which-to-pick)
9. [References](#9-references)

---

## 1. What problem is RAG solving?

Large language models trained on a fixed corpus suffer three structural limits
that no amount of scaling fixes by itself:

1. **Knowledge cutoff** — the model doesn't know anything that happened after
   its training data was frozen.
2. **Hallucination** — when asked about something it doesn't know, the model
   fills the gap with plausible-sounding but unverifiable text.
3. **Provenance** — even when the model is right, you can't point at *which*
   document its answer came from.

**Retrieval-Augmented Generation** (Lewis et al., 2020) addresses all three by
treating the model as a *generator* over an explicit, swappable *retrieval
corpus*. The model no longer needs to memorise the world — it just needs to
read what the retriever surfaces and write a grounded answer.

> Lewis, P. et al. (2020). [*Retrieval-Augmented Generation for
> Knowledge-Intensive NLP Tasks.*](https://arxiv.org/abs/2005.11401) NeurIPS.

The base RAG loop is two stages:

```
question → retriever → top-K passages → generator (LLM) → answer
```

Everything covered below is an evolution of one or both of those stages. The
core question driving the field is: **what does the retriever look like, and
how often does it run?**

---

## 2. Naive RAG

```
retrieve → generate
```

**The pattern.** Embed the question, run cosine similarity against a
pre-embedded chunk index, stuff the top-K chunks into the prompt, call the
LLM once.

**Key insight.** Most factual questions can be answered from a small window
of the corpus that is *semantically close* to the question. Dense vector
similarity (originating with DPR — Karpukhin et al., 2020) is a fast, scalable
way to find that window.

**Why it shines.**
- One LLM call → cheapest and lowest-latency pattern.
- Embeddings can be reused for the whole index lifetime; queries are sub-100 ms
  with HNSW or IVF indexes.
- Trivial to debug: the top-K chunks are the *entire* context the model saw.

**Where it fails.**
- **Vocabulary mismatch**: if the question phrases something differently from
  the source (formulas, abbreviations, jargon), cosine ranks the right chunk
  too low and the model can't answer.
- **Multi-hop reasoning**: any question whose answer requires combining
  facts from two distant chunks degrades, because both chunks must
  individually clear the top-K bar.
- **Negation / absence** queries ("what is *not* in this paper?") — top-K
  retrieves what's similar, never what's missing.
- **No structural awareness**: the retriever can't traverse relationships,
  follow citations, or aggregate across communities of related content.

**Implemented in deepRAG:** [`rag/naive/`](../rag/naive/) — a 2-node LangGraph
(`retrieve → generate`) backed by PGVector cosine search.

---

## 3. Agentic RAG

```
agent ⇄ tools  (until agent emits no tool calls)
```

**The pattern.** Promote the LLM from a passive *consumer* of retrieved
context to an active *driver* of retrieval. The LLM is given a set of tools
(`vector_search`, `kg_lookup`, `web_search`, …) and decides per turn which
tools to call, with what arguments, and when it has enough information.

**Key insight.** A single retrieve-then-generate pass is brittle because
*the model never gets to refine the search*. By making retrieval a tool the
LLM controls — and letting it loop — the model can explore multiple angles,
backtrack, and stop when satisfied.

The mechanism comes from **ReAct** (Yao et al., 2022): interleave
*reasoning* steps with *acting* steps so each acts on the result of the last.
Modern function-calling LLMs (OpenAI, Anthropic, DeepSeek, Llama 3+ Instruct)
implement this natively, and LangGraph's `ToolNode + tools_condition` provides
the routing.

> Yao, S. et al. (2022). [*ReAct: Synergizing Reasoning and Acting in
> Language Models.*](https://arxiv.org/abs/2210.03629) ICLR 2023.

**Why it shines.**
- **Multi-faceted questions**: "compare X and Y" → agent issues two parallel
  searches, one per entity.
- **Exploratory / negation queries**: agent can search "background", "future
  work", "limitations" separately and assemble the answer.
- **Mixed-source retrieval**: agent picks per query whether to call vector
  search vs. graph lookup vs. an external API.
- The agent's own thoughts (visible in the trace) are an audit trail.

**Where it fails.**
- **Cost scales with turns**: each agent turn is one LLM call. Complex
  questions can balloon to 5-10 LLM calls.
- **Latency**: parallel tool calls help, but you're still bottlenecked on
  LLM round-trips.
- **Tool-call hallucination**: the agent can invent plausible-but-empty
  searches and burn turns. Caps (`recursion_limit`) and well-typed tool
  schemas matter.
- **Simple lookups are overkill**: a 100 ms naive query becomes a 5-second
  agent dance.

**Implemented in deepRAG:** [`rag/agentic/`](../rag/agentic/) — a ReAct-style
loop using LangGraph's prebuilt `ToolNode` and the `tools_condition` edge.
Tools: `vector_search(query, k)` over PGVector and
`kg_lookup(entity_name)` over the sqlite-graph KG.

---

## 4. Graph RAG

```
extract_query_entities → expand_subgraph → fetch_chunks → generate
```

**The pattern.** Build a typed knowledge graph from the corpus (entities and
relations extracted by an LLM). At query time, identify the entities the
question mentions, traverse their neighbourhood in the graph, and fetch the
source chunks of every entity and edge the traversal hit. Generate the answer
from the assembled subgraph + chunks.

**Key insight.** Some kinds of questions are answered not by *finding the
right paragraph* but by *understanding how concepts connect*. "Who created
X?" "What depends on Y?" "What does company Z make?" — these are graph queries
in disguise, and a vector index can answer them only when one chunk happens
to contain the whole chain.

Microsoft's GraphRAG (Edge et al., 2024) popularised this approach for
enterprise QA, using LLM extraction + community detection to build the graph
and combining *local* search (one entity's neighbourhood) with *global*
search (community-level summaries) at query time.

> Edge, D. et al. (2024). [*From Local to Global: A Graph RAG Approach to
> Query-Focused Summarization.*](https://arxiv.org/abs/2404.16130) Microsoft
> Research.

**Why it shines.**
- **Relationship questions**: "Who authored the LoopRAG paper?" — follow
  `AUTHORED` edges. "What products does LangChain Inc. maintain?" — follow
  `MAINTAINS` edges. Naive RAG would have to hope the right chunk appears in
  top-K; the graph just *knows*.
- **Provenance is structural**: every relation cites the chunks that
  supported its extraction.
- **Compression**: a noisy long document becomes a compact set of typed
  entities and edges — much easier for the LLM to reason over than raw text.

**Where it fails.**
- **Bounded by extraction quality**: if your extraction prompt didn't capture
  a relation type, no traversal will find it. The KG is only as good as the
  prompt that built it.
- **Negation and absence**: graphs encode what *is*, not what *isn't*. Asking
  "what doesn't X depend on?" can't be answered by traversal at all.
- **Build cost**: one LLM extraction call per chunk during indexing — pricey
  on large corpora.
- **Cypher alpha limitations**: in our case (sqlite-graph v0.1.0-alpha), no
  variable-length paths or aggregations, so we do the BFS in Python.

**Implemented in deepRAG:** [`rag/graph_rag/`](../rag/graph_rag/) — entity
extraction by DeepSeek, 2-hop BFS in Python over `sqlite-graph`, chunk
back-reference via the `source_chunks` array stored on each entity/edge.

---

## 5. Loop RAG (PDCA)

```
plan → do → check ⇄ act   (until converged or max iterations)
```

**The pattern.** A closed-loop multi-agent system that treats answer
generation as a *control problem*. Each iteration:

- **Plan** classifies the user intent and picks a prompt strategy.
- **Do** retrieves evidence (hybrid: vector + graph) and generates a candidate
  answer.
- **Check** scores the answer along **three deviation axes**:
  - `Align(y, q) = 1 − cos(E(y), E(q))` — does the answer match the question?
  - `Faith(y, k) = 1 − mean(δᵢ)` — is the answer grounded in retrieved evidence?
  - `Constraint(y, p) = 1 − Ψ(y, p)` — does the answer satisfy task constraints?
  
  These combine into a composite loss with scenario-dependent weights.
- **Act** attributes the dominant deviation, rewrites the query, and swaps
  the prompt template. The loop closes back to **Plan**.

Termination: total deviation below threshold, *or* max iterations.

**Key insight.** Treat the model's output as a *signal in a feedback loop*
rather than a one-shot artifact. The "three-axis deviation" decomposes
quality into orthogonal dimensions, so the system can identify the *cause* of
a bad answer (off-topic? hallucinating? wrong format?) and apply a targeted
correction in the next iteration.

This is the **PDCA control philosophy** (Plan-Do-Check-Act, Deming, 1950s)
mapped onto LLM reasoning. It also has clear lineage from:

- **Self-RAG** (Asai et al., 2024) — LLM emits reflection tokens to grade its
  own answer and decide whether to retrieve more.
- **CRAG / Corrective RAG** (Yan et al., 2024) — retrieval evaluator triggers
  a corrective web search when retrieved docs are weak.

What LoopRAG (the paper) adds: a *decomposed* deviation signal and an
*attribution* step so corrections are targeted, not random.

> Bai, J. et al. (2026). [*LoopRAG: A Closed-Loop Multi-Agent Retrieval-
> Augmented Generation Framework for Smart Buildings.*](https://doi.org/10.3390/buildings16010196)
> Buildings 16(1), 196.
>
> Asai, A. et al. (2024). [*Self-RAG: Learning to Retrieve, Generate, and
> Critique through Self-Reflection.*](https://arxiv.org/abs/2310.11511) ICLR 2024.
>
> Yan, S. et al. (2024). [*Corrective Retrieval Augmented
> Generation.*](https://arxiv.org/abs/2401.15884) arXiv.

**Why it shines.**
- **Hard / ambiguous questions** where a single retrieval is insufficient:
  the act node can rewrite the query with concepts learned during do/check.
- **Safety-critical domains** (building control, medical, legal) where you
  want a *measurable* quality signal and a refusal path. The faith axis is
  effectively a hallucination detector.
- **Long-horizon reasoning**: each loop accumulates evidence, so the final
  answer reflects 3–4 retrieves' worth of context, not 1.

**Where it fails.**
- **Cost scales linearly with iterations** — N PDCA loops ≈ 4N LLM calls.
  Far slower than naive on simple lookups.
- **Single-direction refinement**: the act node refines the *same* query in
  the *same* direction. It can't laterally explore the corpus the way
  agentic RAG can — see the side-by-side in the project's session log where
  Loop got stuck on a question that agentic solved with multi-angle search.
- **Faith grader brittleness**: if the LLM scoring per-evidence support is
  too strict, the loop can never converge and bottoms out at the iteration
  cap with no answer.

**Implemented in deepRAG:** [`rag/loop/`](../rag/loop/) — `plan_node`,
`make_do_node`, `check_node`, `act_node`, with conditional edge controlled by
total-deviation threshold. State carries `intent`, `prompt_config`,
`evidence`, `answer`, `deviation`, `iteration`, `converged`, and an
append-only `history`. Mirrors the architecture in the Bai et al. paper.

---

## 6. NodeRAG (heterogeneous nodes + PPR)

```
embed_query → vector_seed → ppr_propagate → fetch_chunks → generate
```

**The pattern.** Instead of treating the corpus as either (a) flat chunks
(naive) or (b) entities-and-relations only (Graph RAG), build a
**heterogeneous graph** with *multiple node types*, each carrying its own
embedding:

| Node type            | Created from                              |
|----------------------|-------------------------------------------|
| `entity`             | Named things from KG extraction           |
| `relationship`       | Edges promoted to first-class nodes       |
| `semantic_unit`      | LLM-extracted atomic factual claims       |
| `attribute`          | LLM-extracted entity.name = value triples |
| `high_level_element` | Louvain community labels                  |
| `high_level_overview`| LLM summary per community                 |

At query time, run **two-stage retrieval**:

1. **Shallow (HNSW vector search)** — find top-K seed nodes across all types
   by cosine similarity to the question.
2. **Deep (Personalized PageRank)** — propagate relevance from the seeds
   through the heterogeneous graph; high-PageRank nodes are *structurally*
   relevant even if they don't directly match the query embedding.

Finally, fetch the source chunks of the top-N PPR-ranked nodes and generate.

**Key insight.** Graph RAG conflates "everything about a topic" into one
entity node, losing granularity. Lightweight community-only approaches
(LightRAG, Guo et al., 2024) miss the chunk-level detail. NodeRAG's
contribution is *type-aware* retrieval: a `semantic_unit` carries a single
atomic claim with its own embedding; a `high_level_overview` carries a
community-level summary; both compete for relevance against entities and
relationships. The PPR step then says "given these seeds, what else in the
graph is structurally important?" — surfacing supporting facts the query
wouldn't have hit directly.

> Xu, Y. et al. (2025). *NodeRAG: Structuring Graph-based RAG with
> Heterogeneous Nodes.* (NodeRAG architecture; we implement Xu et al.'s
> six-node-type schema with our own PPR formulation and hybrid candidate
> pool.)
>
> Guo, Z. et al. (2024). [*LightRAG: Simple and Fast Retrieval-Augmented
> Generation.*](https://arxiv.org/abs/2410.05779) arXiv. (Community-only
> alternative; NodeRAG generalises by adding more node types.)

**Personalized PageRank, in one paragraph.** Standard PageRank assigns each
node a score equal to the probability you'd land on it from a random walk
that occasionally teleports to a uniform-random node. *Personalized* PageRank
teleports back to a *biased* set — your seeds — so the resulting scores
measure "structural importance *given* this query's focus." Type weights in
the personalization vector (we boost `semantic_unit` and
`high_level_overview`, downweight bare `relationship`) bias the walk toward
nodes that carry actual factual text rather than purely structural links.

**Hybrid candidate pool.** PPR alone is biased toward chunks tied to
highly-connected entities, which misses formula-heavy or detail-heavy chunks
that have few graph neighbours. Our implementation merges PPR-derived chunks
with a direct cosine search on `rag_chunks`, then re-ranks the combined pool
by cosine to keep only the most query-relevant. This recovers the chunks the
PPR walk skipped.

**Why it shines.**
- **Detail + structure in one pass**: semantic_units carry the actual facts;
  high_level_overviews summarise communities; PPR finds connections.
  Vs. plain Graph RAG, the answer can include both relationships *and*
  numerical detail.
- **Fast at query time** (sub-second on a 3,000-node graph): PPR runs on a
  cached networkx graph; no LLM calls until the final generate step.
- **One unified embedding space**: a single HNSW index serves all six node
  types; node-type filtering is just a `WHERE` clause.

**Where it fails.**
- **Heavy indexing cost**: 2-3 LLM extraction calls per chunk during build,
  plus community summaries. ~$1-2 in tokens and ~15 min for a 7-page paper
  in our setup.
- **Same negation blind spot as Graph RAG**: PPR finds structurally
  *connected* nodes, never explicitly *disconnected* ones.
- **Hyperparameters matter**: top-K seeds, ppr_alpha, chunk_top_k, type
  weights — all interact. We arrived at the current defaults (20 / 0.15 / 18 /
  semantic_unit×1.4) by debugging actual retrieval failures, not theory.

**Implemented in deepRAG:** [`rag/noderag/`](../rag/noderag/) — five-node
LangGraph with the hybrid candidate pool. Builds via
[`scripts/build_noderag.py`](../scripts/build_noderag.py). Stores in a
separate pgvector table `noderag_nodes`; graph topology lives in the same
sqlite-graph DB Graph RAG uses.

---

## 7. Comparison matrix

| Aspect | Naive | Agentic | Graph | Loop (PDCA) | NodeRAG |
|---|---|---|---|---|---|
| **LLM calls per query** | 1 | 3-10 | 2 | 4-16 (1-4 PDCA iters × 4) | 1 |
| **Query latency** (this project, paper corpus) | ~1-7 s | ~5-10 s | ~1-2 s | ~10-20 s | ~1 s |
| **Index build cost** | embed only | embed only | + KG extraction (1 LLM/chunk) | + KG extraction | + KG + 2 LLM/chunk + N community summaries |
| **Index size** | chunks only | chunks + KG | chunks + KG | chunks + KG | chunks + KG + 6-type node table |
| **Tuneable knobs** | top-K | recursion limit, tool set | hops, entity match | max_iters, threshold, weights | top-K seeds, PPR α, chunk top-K, seed-type weights |
| **Best for** | direct factual recall | multi-faceted, exploratory, negation | explicit relationship traversal | hard, ambiguous, safety-critical | heterogeneous corpora, mixed detail + structure |
| **Worst at** | multi-hop, negation, vocabulary mismatch | simple lookups (overkill) | text-only answers (formulas, tables) | simple lookups, lateral search | negation, untuned hyperparameters |
| **Provenance signal** | chunk IDs | tool-call audit trail | matched entities + edges | per-iteration deviation breakdown | seed types + PPR scores + chunk back-refs |
| **Hallucination defence** | none beyond grounding | self-reported in agent thoughts | structural grounding | explicit faith axis | structural + semantic in hybrid pool |

---

## 8. Decision guide — which to pick

Pick by the *shape of the question*, not by the patterns' relative
sophistication. None of these is "best" in general; each wins a specific class
of queries and loses others.

```
                       Is the answer in one paragraph of the corpus?
                                   │
                       ┌───────────┴───────────┐
                       │                       │
                      YES                      NO
                       │                       │
              Are you on a budget?    Does it require following
                       │              relationships across chunks?
              ┌────────┴────────┐               │
              │                 │      ┌────────┴────────┐
            YES                NO     YES                NO
              │                 │      │                 │
           NAIVE            AGENTIC  GRAPH           ┌───┴────┐
                                                     │        │
                                            Numerical /   Exploratory /
                                            ablation /    negation /
                                            convergent?   multi-angle?
                                                 │            │
                                              LOOP        AGENTIC
                                                                │
                                                                or
                                                                │
                                                            NODERAG
                                                  (when you want both
                                                   structure AND detail
                                                   in one fast pass)
```

**Heuristics from this project's empirical session:**

- *"What is X?"* → **naive**.
- *"Who created X and when?"* → **naive** (if a single chunk has both) or **graph** (otherwise).
- *"Compare X and Y."* → **agentic** (issues parallel searches per entity).
- *"What's the relationship between X, Y, Z?"* → **graph**.
- *"Trace the lineage of X through its components."* → **graph** or **noderag**.
- *"Explain X with formulas / give exact numbers."* → **agentic** (multi-angle pulls in formula chunks) or **noderag tuned** (with high `chunk_top_k`).
- *"What does X NOT depend on?"* → **agentic** only; all graph-based patterns fail because absence isn't a graph edge.
- *"Why does X work?"* (open-ended design rationale) → **loop** if the corpus has the rationale explicitly; everyone fails otherwise.
- *"What ablation results did the paper report?"* → **loop** shines because act-node query refinement can pull in the specific component names.

A **hybrid system** would route queries to the best pattern automatically.
That router is itself a small LLM call ahead of retrieval — that's how
production systems usually compose these patterns.

---

## 9. References

The patterns above stand on these papers:

| # | Paper | Year |
|---|-------|------|
| 1 | Lewis et al. [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) | NeurIPS 2020 |
| 2 | Karpukhin et al. [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/abs/2004.04906) | EMNLP 2020 |
| 3 | Yao et al. [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629) | ICLR 2023 |
| 4 | Asai et al. [*Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*](https://arxiv.org/abs/2310.11511) | ICLR 2024 |
| 5 | Yan et al. [*Corrective Retrieval Augmented Generation*](https://arxiv.org/abs/2401.15884) | arXiv 2024 |
| 6 | Edge et al. [*From Local to Global: A Graph RAG Approach to Query-Focused Summarization*](https://arxiv.org/abs/2404.16130) | Microsoft 2024 |
| 7 | Guo et al. [*LightRAG: Simple and Fast Retrieval-Augmented Generation*](https://arxiv.org/abs/2410.05779) | arXiv 2024 |
| 8 | Xu et al. *NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes* | 2025 |
| 9 | Bai et al. [*LoopRAG: A Closed-Loop Multi-Agent RAG Framework*](https://doi.org/10.3390/buildings16010196) | Buildings 2026 |

Plus the underlying control-theory and graph-algorithm work:

- Deming, W. E. *The PDCA cycle* (Plan-Do-Check-Act). 1950s.
- Page, L. et al. *The PageRank Citation Ranking.* 1998. (Personalized variant due to Haveliwala, 2002.)
- Blondel, V. D. et al. [*Fast unfolding of communities in large networks*](https://arxiv.org/abs/0803.0476) (the Louvain algorithm we use for high-level node construction). 2008.
