#!/usr/bin/env bash
# Five-minute interview demo. Runs offline by default; set COHERE_API_KEY to run against the live models.
set -euo pipefail
cd "$(dirname "$0")/.."
export GR_LOG_LEVEL=WARNING
if [ -n "${COHERE_API_KEY:-}" ]; then export GR_OFFLINE=0; MODE="LIVE (Command A / Embed v4 / Rerank 3.5)"; else export GR_OFFLINE=1; MODE="OFFLINE (deterministic backend, no network)"; fi
DB=$(mktemp -u /tmp/grounded-rag-demo-XXXX.sqlite)
step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
show() { python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in sys.argv[1:] if k in d}, indent=2, ensure_ascii=False))' "$@"; }

step "0. mode: $MODE"
step "1. tests + eval gates (recall, MRR, refusal, zero ACL leaks)"
uv run pytest -q 2>&1 | tail -1
uv run python evals/run.py 2>/dev/null
step "2. ingest the sample corpus (6 policy docs with access groups)"
uv run grounded-rag --db "$DB" ingest evals/corpus
step "3. grounded answer with citation offsets (public user)"
uv run grounded-rag --db "$DB" ask "Who can approve business class for a long flight?" | show answer citations
step "4a. HR question as a PUBLIC user -> refusal, nothing leaks"
uv run grounded-rag --db "$DB" ask "How many days of annual leave after 8 years of service?" --groups public | show answer grounded sources
step "4b. same question as HR -> answered"
uv run grounded-rag --db "$DB" ask "How many days of annual leave after 8 years of service?" --groups hr | show answer grounded
step "5. agent mode: tool call -> result -> answer, with the trace and budgets"
uv run grounded-rag --db "$DB" agent "What must a supplier hold before award of a contract involving Protected B information?" | show answer steps tool_calls usage
rm -f "$DB"
printf '\nDone. Code to defend live: src/grounded_rag/retrieval.py (ACL before retrieval, RRF, rerank) and src/grounded_rag/agent.py (budgets, trace).\n'
