from pathlib import Path

import pytest

from rag.core.loader import Chunk, chunk_document, load_documents

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "PG_DSN", "NEO4J_PASSWORD"):
        monkeypatch.setenv(k, "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()


def test_chunk_document_stable_ids():
    text = "alpha. " * 300
    chunks_a = chunk_document(source="t.md", text=text)
    chunks_b = chunk_document(source="t.md", text=text)
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
    assert len(chunks_a) > 1
    assert all(isinstance(c, Chunk) for c in chunks_a)


def test_chunk_document_metadata():
    chunks = chunk_document(source="t.md", text="x" * 2000)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].source == "t.md"


def test_load_documents_markdown(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\n\nWorld.")
    docs = list(load_documents(tmp_path))
    assert len(docs) == 1
    assert docs[0][0].endswith("a.md")
    assert "Hello" in docs[0][1]


def test_load_documents_skips_unsupported(tmp_path):
    (tmp_path / "a.md").write_text("ok")
    (tmp_path / "b.png").write_bytes(b"\x89PNG\r\n")
    docs = list(load_documents(tmp_path))
    assert len(docs) == 1
