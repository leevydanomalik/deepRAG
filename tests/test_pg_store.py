import os

import pytest

from rag.core.config import get_settings
from rag.core.loader import Chunk
from rag.core.pg_store import PgStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    if not os.environ.get("PG_DSN"):
        pytest.skip("PG_DSN not set")
    s = PgStore()
    s.init_schema()
    s.truncate()
    yield s
    s.close()


@pytest.fixture
def dim() -> int:
    return get_settings().embedding_dim


def _fake_chunk(i: int) -> Chunk:
    return Chunk(
        id=f"c{i}",
        source=f"src/{i}.md",
        chunk_index=i,
        text=f"chunk number {i}",
        metadata={"k": "v"},
    )


def test_upsert_and_count(store, dim):
    chunks = [_fake_chunk(i) for i in range(3)]
    embeddings = [[0.1] * dim for _ in chunks]
    store.upsert(chunks, embeddings)
    assert store.count() == 3


def test_similarity_search(store, dim):
    chunks = [_fake_chunk(i) for i in range(3)]
    embeddings = [[float(i)] + [0.0] * (dim - 1) for i in range(3)]
    store.upsert(chunks, embeddings)
    hits = store.similarity_search(query_embedding=[2.0] + [0.0] * (dim - 1), k=2)
    assert len(hits) == 2
    assert hits[0]["id"] == "c2"


def test_upsert_idempotent(store, dim):
    c = _fake_chunk(0)
    store.upsert([c], [[0.0] * dim])
    store.upsert([c], [[0.0] * dim])
    assert store.count() == 1
