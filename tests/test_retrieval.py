from grounded_rag.retrieval import rrf


def test_rrf_prefers_items_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 4, 1]])
    assert fused[1] > fused[2] and fused[3] > fused[4]
    assert fused[1] == fused[3]


def test_hybrid_search_finds_expected_doc(svc):
    hits = svc.rag.retriever.search("How many days before departure must air travel be booked?", groups=["public"], top_n=3)
    assert hits and hits[0].chunk.doc_id == "travel-policy"
    assert hits[0].rerank_score is not None


def test_acl_filters_before_retrieval(svc):
    q = "Within how many hours must a security incident be reported?"
    public_hits = svc.rag.retriever.search(q, groups=["public"], top_n=5)
    assert all(h.chunk.doc_id != "incident-response" for h in public_hits)
    sec_hits = svc.rag.retriever.search(q, groups=["security"], top_n=5)
    assert sec_hits[0].chunk.doc_id == "incident-response"


def test_visible_doc_ids_union_semantics(svc):
    assert "hr-leave" not in svc.store.visible_doc_ids(["public"])
    assert "hr-leave" in svc.store.visible_doc_ids(["hr"])
    assert "travel-policy" in svc.store.visible_doc_ids(["hr"])  # public is always visible
