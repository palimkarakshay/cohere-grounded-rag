"""Hybrid retrieval: dense cosine + BM25, fused with reciprocal rank fusion, then reranked.
Access control is applied BEFORE retrieval (the index only ever sees documents the caller may read),
which is the only placement that cannot leak a chunk through a citation or a tool result."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .backend import Backend, tokenize
from .config import Settings
from .logging import log
from .store import ChunkRecord, Store

logger = logging.getLogger("grounded_rag.retrieval")


@dataclass
class Hit:
    chunk: ChunkRecord
    score: float
    dense_rank: int | None
    lexical_rank: int | None
    rerank_score: float | None = None


def rrf(rankings: list[list[int]], *, k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion over lists of chunk ids (best first)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


class Retriever:
    def __init__(self, store: Store, backend: Backend, settings: Settings):
        self.store, self.backend, self.s = store, backend, settings

    def search(self, query: str, *, groups: list[str], top_n: int | None = None, rerank: bool = True) -> list[Hit]:
        t0 = time.perf_counter()
        visible = self.store.visible_doc_ids(groups)
        chunks, mat = self.store.chunks_for(visible)
        if not chunks:
            log(logger, "search.empty", groups=groups, visible_docs=len(visible))
            return []
        by_id = {c.id: c for c in chunks}
        # dense
        qv = self.backend.embed([query], input_type="search_query")[0]
        norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(qv) or 1.0)
        cos = (mat @ qv) / np.where(norms == 0, 1.0, norms)
        dense_order = [chunks[i].id for i in np.argsort(-cos)[: self.s.dense_k]]
        # lexical
        bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        lex_scores = bm25.get_scores(tokenize(query))
        lex_order = [chunks[i].id for i in np.argsort(-lex_scores)[: self.s.lexical_k] if lex_scores[i] > 0]
        fused = rrf([dense_order, lex_order])
        candidates = sorted(fused.items(), key=lambda kv: -kv[1])
        n = top_n or self.s.rerank_top_n
        hits = [
            Hit(
                chunk=by_id[cid],
                score=score,
                dense_rank=(dense_order.index(cid) + 1) if cid in dense_order else None,
                lexical_rank=(lex_order.index(cid) + 1) if cid in lex_order else None,
            )
            for cid, score in candidates
        ]
        if rerank and hits:
            pool = hits[: max(n * 4, 10)]
            order = self.backend.rerank(query, [h.chunk.text for h in pool], top_n=n)
            reranked = []
            for idx, rs in order:
                if rs <= self.s.min_relevance:
                    continue
                pool[idx].rerank_score = rs
                reranked.append(pool[idx])
            hits = reranked
        else:
            hits = hits[:n]
        log(logger, "search", groups=groups, visible_docs=len(visible), candidates=len(candidates), returned=len(hits),
            ms=round((time.perf_counter() - t0) * 1000))
        return hits
