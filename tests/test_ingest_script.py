from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.ingest import run_ingest


def test_ingest_chunks_and_upserts(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("PG_DSN", "x")
    from rag.core.config import get_settings
    get_settings.cache_clear()

    (tmp_path / "a.md").write_text("# Hello\n\n" + "body text. " * 200)

    fake_store = MagicMock()
    fake_store.upsert.return_value = 0

    with patch("scripts.ingest.PgStore", return_value=fake_store), \
         patch("scripts.ingest.embed_batch", return_value=[[0.0] * 384] * 10):
        n = run_ingest(data_dir=tmp_path)
    assert n > 0
    fake_store.init_schema.assert_called_once()
    assert fake_store.upsert.called
