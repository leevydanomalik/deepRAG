"""Shared prompt templates for all RAG patterns."""

ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the context
below. If the answer is not in the context, say "I don't know based on the provided
context."

Context:
{context}

Question:
{question}

Answer:"""


KG_EXTRACTION_PROMPT = """Extract entities and relations from the text below.
Return STRICT JSON with this exact shape and nothing else:

{{
  "entities": [
    {{"name": "...", "type": "Person|Org|Place|Concept|Product|Event|Other", "description": "..."}}
  ],
  "relations": [
    {{"src": "...", "dst": "...", "type": "...", "description": "..."}}
  ]
}}

Rules:
- Entity names are the canonical proper form (e.g. "LangChain", not "the LangChain library").
- "src" and "dst" must match an entity name in the entities array.
- Keep descriptions under 200 chars.
- Relation types should be SHORT uppercase verbs like CREATED, USES, DEPENDS_ON.

Text:
{text}

JSON:"""


LOOP_PLAN_PROMPT = """You are the PLAN agent in a PDCA RAG loop.
Given the user question, output STRICT JSON:

{{
  "task_type": "factual|explanatory|comparative|predictive|control|diagnosis",
  "entities": ["..."],
  "constraints": ["..."],
  "sub_goals": ["..."],
  "prompt_template": "stuff|stepwise|structured"
}}

Question:
{question}

JSON:"""


LOOP_CHECK_PROMPT = """You are the CHECK agent. Given the question and the proposed answer,
estimate per-evidence support score for each piece of evidence (0..1).
Return STRICT JSON:

{{
  "support_scores": [0.0, 0.0, ...]
}}

Question: {question}
Answer:   {answer}
Evidence (numbered):
{evidence}

JSON:"""


LOOP_ACT_PROMPT = """You are the ACT agent. The previous iteration failed with dominant
deviation '{dominant}'. Suggest a refined query and prompt template.
Return STRICT JSON:

{{
  "rewritten_query": "...",
  "prompt_template": "stuff|stepwise|structured"
}}

Original question: {question}
Previous answer:   {answer}

JSON:"""


GRAPH_RAG_ENTITY_EXTRACT_PROMPT = """Extract the entities (proper nouns, named concepts)
from this question. Return STRICT JSON: {{"entities": ["..."]}}.

Question: {question}

JSON:"""
