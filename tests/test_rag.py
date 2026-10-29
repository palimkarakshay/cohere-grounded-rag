def test_grounded_answer_has_citation_span(svc):
    ans = svc.rag.ask("Who can approve business class for a long flight?", groups=["public"])
    assert ans.grounded and ans.citations
    c = ans.citations[0]
    assert ans.answer[c["start"]:c["end"]] == c["text"]
    assert c["doc_ids"] and c["doc_ids"][0].startswith("travel-policy#")
    assert "Director General" in ans.answer


def test_no_access_means_refusal_not_leak(svc):
    ans = svc.rag.ask("How many days of annual leave after 8 years?", groups=["public"])
    assert not ans.grounded and not ans.sources
    assert "20" not in ans.answer


def test_sources_carry_ranking_signals(svc):
    ans = svc.rag.ask("What is the contract threshold for competition?", groups=["public"])
    assert ans.sources and {"rerank_score", "dense_rank", "lexical_rank"} <= set(ans.sources[0])
