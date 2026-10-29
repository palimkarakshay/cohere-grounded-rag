from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    ordinal: int
    text: str


def chunk_text(doc_id: str, text: str, *, size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Paragraph-aware sliding window. Paragraphs are packed up to `size` chars; windows overlap by
    `overlap` chars so a sentence split across a boundary still appears whole in one chunk."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    buf = ""
    for p in paras:
        if len(p) > size:  # hard-split an oversized paragraph
            if buf:
                chunks.append(Chunk(doc_id, len(chunks), buf))
                buf = ""
            start = 0
            while start < len(p):
                piece = p[start : start + size]
                chunks.append(Chunk(doc_id, len(chunks), piece))
                if start + size >= len(p):
                    break
                start += size - overlap
            continue
        candidate = f"{buf}\n\n{p}" if buf else p
        if len(candidate) <= size:
            buf = candidate
        else:
            chunks.append(Chunk(doc_id, len(chunks), buf))
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n\n{p}".strip() if tail else p
    if buf:
        chunks.append(Chunk(doc_id, len(chunks), buf))
    return chunks
