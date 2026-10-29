"""Grounded question answering: retrieve (permission-aware) -> answer with citations.
The model only ever sees chunks the caller is allowed to read; citations carry chunk and doc ids."""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .backend import Backend
from .config import Settings
from .logging import log
from .retrieval import Hit, Retriever
from .store import Store

logger = logging.getLogger("grounded_rag.rag")

SYSTEM = (
    "You are a careful assistant for an enterprise knowledge base. Answer ONLY from the provided documents. "
    "If the documents do not contain the answer, say so plainly and do not guess. Keep answers short and cite."
)


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    grounded: bool
    usage: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hits_to_documents(hits: list[Hit], store: Store) -> list[dict[str, Any]]:
    docs = []
    for h in hits:
        d = store.get_doc(h.chunk.doc_id)
        docs.append({
            "id": f"{h.chunk.doc_id}#{h.chunk.ordinal}",
            "data": {"title": d.title if d else h.chunk.doc_id, "text": h.chunk.text, "source": d.source if d else ""},
        })
    return docs


class RAG:
    def __init__(self, store: Store, backend: Backend, settings: Settings):
        self.store, self.backend, self.s = store, backend, settings
        self.retriever = Retriever(store, backend, settings)

    def ask(self, question: str, *, groups: list[str], top_n: int | None = None) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, groups=groups, top_n=top_n)
        docs = hits_to_documents(hits, self.store)
        if not docs:
            return Answer(question, "No documents you have access to cover this question.", [], [], grounded=False,
                          backend=self.backend.name, latency_ms=round((time.perf_counter() - t0) * 1000))
        res = self.backend.chat([{"role": "user", "content": question}], documents=docs, system=SYSTEM)
        cits = [{"start": c.start, "end": c.end, "text": c.text, "doc_ids": c.doc_ids} for c in res.citations]
        sources = [{"id": d["id"], "title": d["data"]["title"], "source": d["data"]["source"],
                    "rerank_score": h.rerank_score, "dense_rank": h.dense_rank, "lexical_rank": h.lexical_rank}
                   for d, h in zip(docs, hits, strict=True)]
        # grounded = the model answered from the documents (citations present); an explicit "not covered" is not grounded
        ans = Answer(question, res.text, cits, sources, grounded=bool(cits), usage=res.usage,
                     backend=self.backend.name, latency_ms=round((time.perf_counter() - t0) * 1000))
        log(logger, "ask", groups=groups, sources=len(sources), citations=len(cits), grounded=ans.grounded, ms=ans.latency_ms)
        return ans
