"""Wiring: one Service object owns the store, backend, RAG and agent. Ingest lives here."""
from __future__ import annotations

import logging
from pathlib import Path

from .agent import Agent
from .backend import make_backend
from .chunking import chunk_text
from .config import Settings, get_settings
from .logging import configure, log
from .rag import RAG
from .store import DocRecord, Store

logger = logging.getLogger("grounded_rag.service")


class Service:
    def __init__(self, settings: Settings | None = None, *, db_path: str | None = None):
        self.s = settings or get_settings()
        configure(self.s.log_level)
        self.backend = make_backend(self.s)
        self.store = Store(db_path or self.s.db_path)
        self.rag = RAG(self.store, self.backend, self.s)
        self.agent = Agent(self.store, self.backend, self.s)
        log(logger, "service.start", backend=self.backend.name, db=db_path or self.s.db_path)

    def ingest(self, doc_id: str, title: str, text: str, *, source: str = "", groups: list[str] | None = None) -> int:
        groups = groups or ["public"]
        self.store.upsert_doc(DocRecord(doc_id, title, text, source, groups))
        chunks = chunk_text(doc_id, text, size=self.s.chunk_chars, overlap=self.s.chunk_overlap)
        texts = [c.text for c in chunks]
        emb = self.backend.embed(texts, input_type="search_document")
        self.store.add_chunks(doc_id, texts, emb)
        log(logger, "ingest", doc_id=doc_id, chunks=len(chunks), groups=groups)
        return len(chunks)

    def ingest_dir(self, path: str | Path) -> int:
        """Ingest every .md/.txt in a directory. Front-matter line `groups: a, b` sets access groups."""
        n = 0
        for f in sorted(Path(path).glob("*")):
            if f.suffix not in {".md", ".txt"}:
                continue
            raw = f.read_text(encoding="utf-8")
            groups, title, body = parse_frontmatter(raw, default_title=f.stem)
            n += self.ingest(f.stem, title, body, source=str(f), groups=groups)
        return n


def parse_frontmatter(raw: str, *, default_title: str) -> tuple[list[str], str, str]:
    groups, title = ["public"], default_title
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            for line in raw[3:end].strip().splitlines():
                k, _, v = line.partition(":")
                if k.strip() == "groups":
                    groups = [g.strip() for g in v.split(",") if g.strip()]
                elif k.strip() == "title":
                    title = v.strip()
            raw = raw[end + 4 :].lstrip("\n")
    return groups, title, raw
