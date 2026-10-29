from grounded_rag.config import Settings
from grounded_rag.service import Service

from .conftest import CORPUS


def test_agent_uses_tool_then_answers(svc):
    run = svc.agent.run("Who approves a sole-source justification above 40,000 dollars?", groups=["public"])
    kinds = [s.kind for s in run.steps]
    assert kinds[0] == "tool_call" and kinds[-1] == "answer"
    assert run.tool_calls == 1
    assert "Assistant Deputy Minister" in run.answer


def test_agent_tools_respect_acl(svc):
    run = svc.agent.run("Within how many hours must a security incident be reported?", groups=["public"])
    assert run.steps[0].kind == "tool_call"
    assert "incident-response" not in run.steps[0].result_summary
    assert "one hour" not in run.answer.lower()


def test_agent_tool_budget_stops_loop():
    s = Service(Settings(offline=True, agent_max_tool_calls=0, log_level="WARNING"), db_path=":memory:")
    s.ingest_dir(CORPUS)
    run = s.agent.run("anything", groups=["public"])
    assert run.steps[-1].kind == "budget_exhausted" and "budget" in run.answer


def test_get_document_tool_denies_hidden_doc(svc):
    from grounded_rag.agent import ToolBox

    box = ToolBox(svc.store, svc.agent.retriever, ["public"])
    assert "error" in box.get_document("hr-leave")
    assert box.get_document("travel-policy")["results"][0]["doc_id"] == "travel-policy"
