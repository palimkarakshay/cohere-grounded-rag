"""LLM backend abstraction: one Protocol, a live Cohere V2 implementation, and a deterministic
offline implementation so the whole system (API, agent loop, evals) runs and tests without network.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .config import Settings
from .logging import log

logger = logging.getLogger("grounded_rag.backend")

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how", "in", "is", "it", "its", "may", "must", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when", "where", "which", "who", "will", "with", "within", "you", "your"])


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_terms(text: str) -> set[str]:
    """Query terms that carry meaning: no stop words, no 1-char tokens. Used by the offline backend only."""
    return {t for t in tokenize(text) if t not in _STOP and len(t) > 1}


@dataclass
class Citation:
    start: int
    end: int
    text: str
    doc_ids: list[str]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    text: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_plan: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw_message: dict[str, Any] | None = None  # assistant message to append to history for tool loops


class Backend(Protocol):
    name: str

    def embed(self, texts: list[str], *, input_type: str) -> np.ndarray: ...

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]]: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        documents: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> ChatResult: ...


# ----------------------------------------------------------------------------------------------
# Live Cohere backend (ClientV2)
# ----------------------------------------------------------------------------------------------
class CohereBackend:
    name = "cohere"

    def __init__(self, settings: Settings):
        import cohere  # imported lazily so the offline path never needs the SDK's network stack

        self.s = settings
        self.co = cohere.ClientV2(api_key=settings.cohere_api_key)

    def embed(self, texts: list[str], *, input_type: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 96):  # API limit: 96 texts per call
            batch = texts[i : i + 96]
            t0 = time.perf_counter()
            resp = self.co.embed(
                model=self.s.embed_model,
                texts=batch,
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self.s.embed_dim,
            )
            out.extend(resp.embeddings.float_)
            log(logger, "embed", model=self.s.embed_model, n=len(batch), ms=round((time.perf_counter() - t0) * 1000))
        return np.asarray(out, dtype=np.float32)

    def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        t0 = time.perf_counter()
        resp = self.co.rerank(model=self.s.rerank_model, query=query, documents=documents, top_n=top_n)
        log(logger, "rerank", model=self.s.rerank_model, n=len(documents), top_n=top_n, ms=round((time.perf_counter() - t0) * 1000))
        return [(r.index, float(r.relevance_score)) for r in resp.results]

    def chat(self, messages, *, documents=None, tools=None, system=None) -> ChatResult:
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        kwargs: dict[str, Any] = {"model": self.s.chat_model, "messages": msgs, "temperature": 0.2}
        if documents:
            kwargs["documents"] = documents
        if tools:
            kwargs["tools"] = tools
        t0 = time.perf_counter()
        resp = self.co.chat(**kwargs)
        m = resp.message
        text = "".join(c.text for c in (m.content or []) if getattr(c, "type", "text") == "text")
        cits: list[Citation] = []
        for c in m.citations or []:
            ids = []
            for src in c.sources or []:
                doc = getattr(src, "document", None)
                did = (doc or {}).get("id") if isinstance(doc, dict) else getattr(src, "id", None)
                if did:
                    ids.append(str(did))
            cits.append(Citation(start=c.start or 0, end=c.end or 0, text=c.text or "", doc_ids=ids))
        calls = [
            ToolCall(id=tc.id or uuid.uuid4().hex, name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (m.tool_calls or [])
        ]
        usage = {}
        if resp.usage and resp.usage.billed_units:
            usage = {"input_tokens": resp.usage.billed_units.input_tokens, "output_tokens": resp.usage.billed_units.output_tokens}
        raw = m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else None
        log(logger, "chat", model=self.s.chat_model, tool_calls=len(calls), citations=len(cits), usage=usage, ms=round((time.perf_counter() - t0) * 1000))
        return ChatResult(text=text, citations=cits, tool_calls=calls, tool_plan=m.tool_plan, usage=usage, raw_message=raw)


# ----------------------------------------------------------------------------------------------
# Offline backend: deterministic, no network. Hashed bag-of-words embeddings, overlap reranking,
# extractive grounded answers with real citation spans, and a scripted tool-use policy.
# ----------------------------------------------------------------------------------------------
class OfflineBackend:
    name = "offline"

    def __init__(self, settings: Settings):
        self.s = settings
        self.dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts, *, input_type):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def rerank(self, query, documents, *, top_n):
        q = content_terms(query)
        scored = []
        for i, d in enumerate(documents):
            toks = tokenize(d)
            if not toks or not q:
                scored.append((i, 0.0))
                continue
            overlap = len(q & set(toks))
            scored.append((i, overlap / math.sqrt(len(toks)) / len(q)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]

    def chat(self, messages, *, documents=None, tools=None, system=None) -> ChatResult:
        user_text = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content")
                user_text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c)
        has_tool_results = any(m.get("role") == "tool" for m in messages)
        if tools and not has_tool_results:
            tool_names = [t["function"]["name"] for t in tools]
            name = "search_documents" if "search_documents" in tool_names else tool_names[0]
            call = ToolCall(id="call_" + hashlib.md5(user_text.encode()).hexdigest()[:8], name=name, arguments={"query": user_text})
            raw = {"role": "assistant", "tool_plan": "Search the knowledge base first.", "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}]}
            return ChatResult(text="", tool_calls=[call], tool_plan=raw["tool_plan"], raw_message=raw)
        docs = list(documents or [])
        if has_tool_results and not docs:
            for m in messages:
                if m.get("role") == "tool":
                    for item in m.get("content", []):
                        try:
                            payload = json.loads(item["document"]["data"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        for r in payload.get("results", []):
                            docs.append({"id": r.get("id"), "data": {"title": r.get("title", ""), "text": r.get("text", "")}})
        if not docs:
            return ChatResult(text="I could not find any relevant documents to answer that.", citations=[])
        q = content_terms(user_text)
        best = None
        for d in docs:
            text = d["data"].get("text", "")
            for sent in re.split(r"(?<=[.!?])\s+", text):
                score = len(q & set(tokenize(sent)))
                if best is None or score > best[0] or (score == best[0] and len(sent) < len(best[1])):
                    best = (score, sent.strip(), d["id"])
        score, sent, did = best
        if score < 2:  # a real model declines when the documents do not cover the question; so do we
            return ChatResult(text="The provided documents do not contain the answer to that question.", citations=[],
                              raw_message={"role": "assistant", "content": "not covered"})
        answer = f"According to the documents: {sent}"
        start = answer.index(sent)
        cit = Citation(start=start, end=start + len(sent), text=sent, doc_ids=[did])
        return ChatResult(text=answer, citations=[cit], raw_message={"role": "assistant", "content": answer})


def make_backend(settings: Settings) -> Backend:
    if settings.use_live:
        return CohereBackend(settings)
    return OfflineBackend(settings)
