"""SQLite document store with per-document access groups and numpy chunk embeddings.
Small by design: a single file, no server, works air-gapped. Swap for pgvector/Qdrant at scale."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass
class DocRecord:
    id: str
    title: str
    text: str
    source: str
    groups: list[str]


@dataclass
class ChunkRecord:
    id: int
    doc_id: str
    ordinal: int
    text: str


class Store:
    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        c = self.conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs(
              id TEXT PRIMARY KEY, title TEXT NOT NULL, text TEXT NOT NULL,
              source TEXT NOT NULL DEFAULT '', groups_json TEXT NOT NULL DEFAULT '["public"]');
            CREATE TABLE IF NOT EXISTS chunks(
              id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
              ordinal INTEGER NOT NULL, text TEXT NOT NULL, embedding BLOB, dim INTEGER);
            CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
            """
        )
        c.commit()

    def upsert_doc(self, doc: DocRecord) -> None:
        self.conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc.id,))
        self.conn.execute(
            "INSERT OR REPLACE INTO docs(id,title,text,source,groups_json) VALUES(?,?,?,?,?)",
            (doc.id, doc.title, doc.text, doc.source, json.dumps(sorted(set(doc.groups)))),
        )
        self.conn.commit()

    def add_chunks(self, doc_id: str, texts: list[str], embeddings: np.ndarray) -> list[int]:
        ids = []
        for i, (t, e) in enumerate(zip(texts, embeddings, strict=True)):
            vec = np.asarray(e, dtype=np.float32)
            cur = self.conn.execute(
                "INSERT INTO chunks(doc_id,ordinal,text,embedding,dim) VALUES(?,?,?,?,?)",
                (doc_id, i, t, vec.tobytes(), int(vec.shape[0])),
            )
            ids.append(int(cur.lastrowid))
        self.conn.commit()
        return ids

    def get_doc(self, doc_id: str) -> DocRecord | None:
        r = self.conn.execute("SELECT * FROM docs WHERE id=?", (doc_id,)).fetchone()
        return DocRecord(r["id"], r["title"], r["text"], r["source"], json.loads(r["groups_json"])) if r else None

    def list_docs(self) -> list[DocRecord]:
        rows = self.conn.execute("SELECT * FROM docs ORDER BY id").fetchall()
        return [DocRecord(r["id"], r["title"], r["text"], r["source"], json.loads(r["groups_json"])) for r in rows]

    def visible_doc_ids(self, groups: list[str]) -> set[str]:
        """Documents the caller may see: any overlap between caller groups and doc groups."""
        allowed = set(groups) | {"public"}
        out = set()
        for r in self.conn.execute("SELECT id, groups_json FROM docs"):
            if allowed & set(json.loads(r["groups_json"])):
                out.add(r["id"])
        return out

    def chunks_for(self, doc_ids: set[str]) -> tuple[list[ChunkRecord], np.ndarray]:
        if not doc_ids:
            return [], np.zeros((0, 0), dtype=np.float32)
        q = f"SELECT id,doc_id,ordinal,text,embedding,dim FROM chunks WHERE doc_id IN ({','.join('?' * len(doc_ids))}) ORDER BY id"
        rows = self.conn.execute(q, tuple(sorted(doc_ids))).fetchall()
        recs = [ChunkRecord(r["id"], r["doc_id"], r["ordinal"], r["text"]) for r in rows]
        if not rows:
            return recs, np.zeros((0, 0), dtype=np.float32)
        mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32, count=r["dim"]) for r in rows])
        return recs, mat

    def count(self) -> dict[str, int]:
        d = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        c = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"docs": d, "chunks": c}
