from grounded_rag.chunking import chunk_text


def test_small_doc_is_one_chunk():
    c = chunk_text("d", "hello world", size=100, overlap=10)
    assert len(c) == 1 and c[0].text == "hello world" and c[0].ordinal == 0


def test_paragraphs_pack_and_overlap():
    paras = [f"para {i} " + "x" * 80 for i in range(10)]
    chunks = chunk_text("d", "\n\n".join(paras), size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(ch.text) <= 200 for ch in chunks)
    # overlap: tail of chunk n appears at the head of chunk n+1
    assert chunks[0].text[-40:] in chunks[1].text


def test_oversized_paragraph_is_hard_split():
    chunks = chunk_text("d", "y" * 1000, size=300, overlap=50)
    assert len(chunks) >= 4 and "".join(c.text[:250] for c in chunks).startswith("y" * 250)


def test_bad_params():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("d", "x", size=10, overlap=10)
