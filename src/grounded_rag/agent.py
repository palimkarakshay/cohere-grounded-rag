"""A small tool-using agent on Cohere V2 tool calling, with the controls a production deployment needs:
a step budget, a tool-call budget, permission-aware tools, and a full trace of every step."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .backend import Backend, ToolCall
from .config import Settings
from .logging import log
from .retrieval import Retriever
from .store import Store

logger = logging.getLogger("grounded_rag.agent")

SYSTEM = (
    "You are an assistant for an enterprise knowledge base. Use the tools to find facts before answering. "
    "Never answer from memory when a tool can check. If nothing relevant is found, say so."
)


@dataclass
class Step:
    kind: str  # "tool_call" | "tool_result" | "answer" | "budget_exhausted"
    name: str | None = None
    arguments: dict[str, Any] | None = None
    result_summary: str | None = None
    text: str | None = None
    ms: int = 0


@dataclass
class AgentRun:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    tool_calls: int = 0
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    backend: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolBox:
    """Tools are closures over the caller's groups, so a tool can never read past the caller's access."""

    def __init__(self, store: Store, retriever: Retriever, groups: list[str]):
        self.store, self.retriever, self.groups = store, retriever, groups

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {
                "name": "search_documents",
                "description": "Search the knowledge base. Returns the most relevant passages with document ids.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}},
                               "required": ["query"]}}},
            {"type": "function", "function": {
                "name": "get_document",
                "description": "Fetch the full text of one document by id (as returned by search_documents, before '#').",
                "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
        ]

    def handlers(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {"search_documents": self.search_documents, "get_document": self.get_document}

    def search_documents(self, query: str) -> dict[str, Any]:
        hits = self.retriever.search(query, groups=self.groups, top_n=4)
        results = []
        for h in hits:
            d = self.store.get_doc(h.chunk.doc_id)
            results.append({"id": f"{h.chunk.doc_id}#{h.chunk.ordinal}", "doc_id": h.chunk.doc_id,
                            "title": d.title if d else "", "text": h.chunk.text, "score": h.rerank_score})
        return {"results": results}

    def get_document(self, doc_id: str) -> dict[str, Any]:
        if doc_id not in self.store.visible_doc_ids(self.groups):
            return {"error": "not found or not permitted", "doc_id": doc_id}
        d = self.store.get_doc(doc_id)
        return {"results": [{"id": d.id, "doc_id": d.id, "title": d.title, "text": d.text[:6000]}]} if d else {"error": "not found"}


class Agent:
    def __init__(self, store: Store, backend: Backend, settings: Settings):
        self.store, self.backend, self.s = store, backend, settings
        self.retriever = Retriever(store, backend, settings)

    def run(self, question: str, *, groups: list[str]) -> AgentRun:
        t0 = time.perf_counter()
        box = ToolBox(self.store, self.retriever, groups)
        handlers = box.handlers()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        run = AgentRun(question=question, answer="", backend=self.backend.name)
        for step in range(self.s.agent_max_steps):
            res = self.backend.chat(messages, tools=box.specs(), system=SYSTEM)
            for k in ("input_tokens", "output_tokens"):
                if k in res.usage:
                    run.usage[k] = run.usage.get(k, 0) + (res.usage[k] or 0)
            if not res.tool_calls:
                run.answer = res.text
                run.citations = [{"start": c.start, "end": c.end, "text": c.text, "doc_ids": c.doc_ids} for c in res.citations]
                run.steps.append(Step(kind="answer", text=res.text))
                break
            messages.append(res.raw_message or {"role": "assistant", "tool_plan": res.tool_plan, "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in res.tool_calls]})
            for tc in res.tool_calls:
                if run.tool_calls >= self.s.agent_max_tool_calls:
                    run.steps.append(Step(kind="budget_exhausted", name=tc.name))
                    run.answer = "Stopped: tool-call budget exhausted before an answer was reached."
                    return self._finish(run, t0, groups)
                out = self._call(tc, handlers)
                run.tool_calls += 1
                run.steps.append(Step(kind="tool_call", name=tc.name, arguments=tc.arguments, ms=out.pop("_ms", 0),
                                      result_summary=_summarize(out)))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": [{"type": "document", "document": {"data": json.dumps(out)}}]})
        else:
            run.answer = run.answer or "Stopped: step budget exhausted before an answer was reached."
            run.steps.append(Step(kind="budget_exhausted"))
        return self._finish(run, t0, groups)

    def _call(self, tc: ToolCall, handlers) -> dict[str, Any]:
        t0 = time.perf_counter()
        fn = handlers.get(tc.name)
        if fn is None:
            out: dict[str, Any] = {"error": f"unknown tool {tc.name}"}
        else:
            try:
                out = fn(**tc.arguments)
            except TypeError as e:
                out = {"error": f"bad arguments: {e}"}
        out["_ms"] = round((time.perf_counter() - t0) * 1000)
        return out

    def _finish(self, run: AgentRun, t0: float, groups: list[str]) -> AgentRun:
        run.latency_ms = round((time.perf_counter() - t0) * 1000)
        log(logger, "agent.run", groups=groups, steps=len(run.steps), tool_calls=run.tool_calls, usage=run.usage, ms=run.latency_ms)
        return run


def _summarize(out: dict[str, Any]) -> str:
    if "error" in out:
        return f"error: {out['error']}"
    res = out.get("results", [])
    return f"{len(res)} result(s): " + ", ".join(r.get("id", "?") for r in res[:4])
