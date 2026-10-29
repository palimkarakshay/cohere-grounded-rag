"""Evaluation runner. Measures the two things that matter for grounded QA in a permissioned corpus:
retrieval quality (Recall@k, MRR over expected documents) and answer behaviour (must-contain hits,
citation presence, and refusal when the caller has no access). Runs offline by default; set
COHERE_API_KEY (and GR_OFFLINE=0) to evaluate the live models. Writes evals/report.md."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from grounded_rag.config import Settings
from grounded_rag.service import Service


def run(k: int = 5) -> dict:
    svc = Service(Settings(log_level="WARNING"), db_path=":memory:")
    svc.ingest_dir(HERE / "corpus")
    rows = [json.loads(line) for line in (HERE / "golden.jsonl").read_text().splitlines() if line.strip()]
    results = []
    for g in rows:
        t0 = time.perf_counter()
        hits = svc.rag.retriever.search(g["question"], groups=g["groups"], top_n=k)
        ranked_docs: list[str] = []
        for h in hits:
            if h.chunk.doc_id not in ranked_docs:
                ranked_docs.append(h.chunk.doc_id)
        expected = set(g["expected_docs"])
        recall = (len(expected & set(ranked_docs[:k])) / len(expected)) if expected else None
        rr = 0.0
        for i, d in enumerate(ranked_docs, start=1):
            if d in expected:
                rr = 1.0 / i
                break
        ans = svc.rag.ask(g["question"], groups=g["groups"], top_n=k, hits=hits)
        contains = all(m.lower() in ans.answer.lower() for m in g["must_contain"]) if g["must_contain"] else None
        refusal_ok = (not ans.grounded) if g.get("expect_refusal") else None
        leaked = bool(set(ranked_docs) - svc.store.visible_doc_ids(g["groups"]))
        results.append({
            "id": g["id"], "recall_at_k": recall, "rr": rr if expected else None, "must_contain": contains,
            "cited": bool(ans.citations), "refusal_ok": refusal_ok, "acl_leak": leaked,
            "ms": round((time.perf_counter() - t0) * 1000), "answer": ans.answer[:160],
        })
    scored = [r for r in results if r["recall_at_k"] is not None]
    summary = {
        "backend": svc.backend.name, "n": len(results), "k": k,
        "recall_at_k": round(sum(r["recall_at_k"] for r in scored) / len(scored), 3),
        "mrr": round(sum(r["rr"] for r in scored) / len(scored), 3),
        "must_contain_rate": round(sum(1 for r in results if r["must_contain"]) / max(1, sum(1 for r in results if r["must_contain"] is not None)), 3),
        "cited_rate": round(sum(1 for r in scored if r["cited"]) / len(scored), 3),
        "refusals_correct": all(r["refusal_ok"] for r in results if r["refusal_ok"] is not None),
        "acl_leaks": sum(1 for r in results if r["acl_leak"]),
        "p50_ms": sorted(r["ms"] for r in results)[len(results) // 2],
    }
    return {"summary": summary, "results": results}


def write_report(out: dict, path: Path) -> None:
    s = out["summary"]
    lines = ["# grounded-rag eval report\n", f"backend: **{s['backend']}** · n={s['n']} · k={s['k']}\n",
             "| metric | value |", "|---|---|"]
    for key in ("recall_at_k", "mrr", "must_contain_rate", "cited_rate", "refusals_correct", "acl_leaks", "p50_ms"):
        lines.append(f"| {key} | {s[key]} |")
    lines += ["", "| id | recall@k | rr | must_contain | cited | refusal_ok | acl_leak | ms | answer |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out["results"]:
        lines.append(f"| {r['id']} | {r['recall_at_k']} | {r['rr']} | {r['must_contain']} | {r['cited']} | {r['refusal_ok']} | {r['acl_leak']} | {r['ms']} | {r['answer'].replace('|', '/')} |")
    lines += ["", "Gates: acl_leaks must be 0; refusals_correct must be true; recall@k >= 0.9."]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    out = run()
    write_report(out, HERE / "report.md")
    print(json.dumps(out["summary"], indent=2))
    s = out["summary"]
    ok = s["acl_leaks"] == 0 and s["refusals_correct"] and s["recall_at_k"] >= 0.9
    sys.exit(0 if ok else 1)
