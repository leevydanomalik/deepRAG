"""Document loading + chunking. Supported: .txt, .md, .pdf."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Import from the character submodule directly to avoid pulling in
# langchain_text_splitters/__init__.py's sentence_transformers chain, which
# breaks if an incompatible transformers/torch combo is installed.
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    chunk_index: int
    text: str
    metadata: dict


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(root: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (absolute_path, full_text) for every supported file under root."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        yield (str(path.resolve()), text)


def chunk_document(
    source: str,
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    extra_metadata: dict | None = None,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    out: list[Chunk] = []
    for i, ct in enumerate(raw_chunks):
        cid = hashlib.sha256(f"{source}:{i}".encode()).hexdigest()
        out.append(
            Chunk(
                id=cid,
                source=source,
                chunk_index=i,
                text=ct,
                metadata={"source": source, **(extra_metadata or {})},
            )
        )
    return out
