"""FastAPI surface. Caller identity/groups arrive in the X-Groups header (comma separated); in a real
deployment this comes from the identity provider, never from the client."""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Header, Request
from pydantic import BaseModel, Field

from . import __version__
from .logging import log, new_request_id
from .service import Service

logger = logging.getLogger("grounded_rag.api")


class IngestRequest(BaseModel):
    id: str
    title: str
    text: str
    source: str = ""
    groups: list[str] = Field(default_factory=lambda: ["public"])


class AskRequest(BaseModel):
    question: str
    top_n: int | None = None


def _groups(header: str | None) -> list[str]:
    return [g.strip() for g in (header or "public").split(",") if g.strip()]


def create_app(service: Service | None = None) -> FastAPI:
    svc = service or Service()
    app = FastAPI(title="grounded-rag", version=__version__)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid = new_request_id()
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["x-request-id"] = rid
        log(logger, "http", method=request.method, path=request.url.path, status=resp.status_code,
            ms=round((time.perf_counter() - t0) * 1000))
        return resp

    @app.get("/health")
    def health():
        return {"ok": True, "backend": svc.backend.name, "version": __version__, **svc.store.count()}

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        n = svc.ingest(req.id, req.title, req.text, source=req.source, groups=req.groups)
        return {"id": req.id, "chunks": n}

    @app.post("/ask")
    def ask(req: AskRequest, x_groups: str | None = Header(default=None)):
        return svc.rag.ask(req.question, groups=_groups(x_groups), top_n=req.top_n).to_dict()

    @app.post("/agent")
    def agent(req: AskRequest, x_groups: str | None = Header(default=None)):
        return svc.agent.run(req.question, groups=_groups(x_groups)).to_dict()

    return app


app = create_app() if __name__ == "grounded_rag.api_autoload" else None
