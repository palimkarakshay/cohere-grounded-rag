# grounded-rag

Permission-aware grounded RAG and a budgeted tool-using agent on the **Cohere API** (Command A, Embed v4, Rerank 3.5), written to be deployed where the data lives: one SQLite file, no network dependency except the model endpoint, an offline backend so every test and eval runs air-gapped, and structured JSON logs on every call.

Built as a small, defensible reference for the kind of system a forward-deployed engineer ships into a regulated enterprise or a government department.

## What it does

```
POST /ingest   doc + access groups  ->  chunk -> embed -> store
POST /ask      question             ->  ACL filter -> dense + BM25 -> RRF -> rerank -> grounded answer with citations
POST /agent    question             ->  tool-calling loop (search_documents, get_document) with step/tool budgets and a trace
GET  /health
```

Design choices that matter in production:

- **Access control before retrieval.** The caller's groups (from the `X-Groups` header; in production from the identity provider) filter the document set *before* the index is searched. A chunk the caller cannot read can never appear in a citation or a tool result. The eval reports `acl_leaks` and gates on zero.
- **Hybrid retrieval.** Dense cosine (Embed v4) and BM25 are fused with reciprocal rank fusion, then Rerank 3.5 orders the top pool. Each source in the response carries its dense rank, lexical rank and rerank score, so a bad answer can be debugged from the log.
- **Grounded answers with real citation spans.** Command A's `documents` mode returns `start`/`end` offsets into the answer plus the document ids; the API returns them verbatim.
- **Agent with budgets and a trace.** Max steps, max tool calls, every tool call and result summarized in `steps[]`. Tools are closures over the caller's groups, so a tool cannot read past the caller's access.
- **Offline backend.** `GR_OFFLINE=1` (or no API key) swaps in a deterministic backend: hashed bag-of-words embeddings, overlap reranking, extractive grounded answers with correct citation spans, and a scripted tool-use policy. The whole system, including the agent loop and the eval, runs without network. Tests use it. Live runs use the real models.
- **Structured logs.** One JSON line per embed/rerank/chat/search/http event with latency, counts, token usage and a request id.

## Demo

`demo/demo.sh` runs the whole story in about a minute: tests and eval gates, ingest, a cited answer, a refusal for a caller without access, the same question answered for a caller with access, and an agent run with its trace. Offline by default; with `COHERE_API_KEY` set it runs against the live models.

## Run

```bash
uv sync --extra dev
uv run pytest                      # offline, ~1s
uv run python evals/run.py         # offline eval -> evals/report.md

# live
export COHERE_API_KEY=...          # https://dashboard.cohere.com
uv run grounded-rag --db kb.sqlite ingest evals/corpus
uv run grounded-rag --db kb.sqlite ask "Who approves a sole-source justification above 40,000 dollars?"
uv run grounded-rag --db kb.sqlite agent "What must a supplier hold before a Protected B contract?" --groups public
uv run grounded-rag --db kb.sqlite serve --port 8080
curl -s localhost:8080/ask -H 'content-type: application/json' -H 'X-Groups: hr' \
     -d '{"question":"How many days of annual leave after 8 years?"}'
GR_OFFLINE=0 uv run python evals/run.py   # live eval
```

Trial keys allow 10 calls per minute: set `GR_RATE_LIMIT_PER_MIN=10` (the client throttles and retries on 429), and expect the live eval to take about four minutes. Latencies in a throttled run include the waits.

Models default to `command-a-03-2025`, `embed-v4.0`, `rerank-v3.5`; override with `GR_CHAT_MODEL`, `GR_EMBED_MODEL`, `GR_RERANK_MODEL` (and `GR_EMBED_DIM`). Private deployments: point the SDK at your endpoint via the usual Cohere client settings.

## Evaluation

`evals/golden.jsonl` holds questions over a small fictional public-sector corpus (`evals/corpus/`, six documents with access groups). The runner measures:

| metric | why |
|---|---|
| `recall_at_k`, `mrr` | did retrieval find the right document, and how high |
| `must_contain_rate` | did the answer contain the fact |
| `cited_rate` | did a grounded answer carry citations |
| `refusals_correct` | when the caller has no access, did we refuse instead of answer |
| `acl_leaks` | did any retrieved chunk come from a document the caller cannot see (must be 0) |
| `p50_ms` | latency |

It exits non-zero when a gate fails, so it works as a CI check. Adding an LLM-as-judge for answer quality is the obvious next step; the golden set already carries the fields it needs.

## Layout

```
src/grounded_rag/
  backend.py    Backend protocol; CohereBackend (ClientV2); OfflineBackend
  chunking.py   paragraph-aware sliding window
  store.py      SQLite docs + chunks + embeddings; group-based visibility
  retrieval.py  dense + BM25 -> RRF -> rerank (ACL applied first)
  rag.py        grounded answer with citations
  agent.py      tool-use loop with budgets and trace
  service.py    wiring + ingest
  api.py        FastAPI
  cli.py        ingest / ask / agent / serve
evals/          corpus, golden set, runner, report
tests/          offline, ~15 tests
```

## Limits

- The store is SQLite plus numpy; it is for tens of thousands of chunks, not millions. Swap `store.py` for pgvector or Qdrant.
- BM25 is rebuilt per query over the visible chunks; cache it per group set for real traffic.
- Citations are only as good as the model's; the eval checks presence and ACL, not citation precision. A judge is the next step.

MIT. Author: Akshay Palimkar.
