from fastapi.testclient import TestClient

from grounded_rag.api import create_app


def test_api_roundtrip(svc):
    client = TestClient(create_app(svc))
    h = client.get("/health").json()
    assert h["ok"] and h["backend"] == "offline" and h["docs"] == 6
    r = client.post("/ingest", json={"id": "x", "title": "X", "text": "The widget code is 4242.", "groups": ["eng"]})
    assert r.status_code == 200 and r.json()["chunks"] == 1
    denied = client.post("/ask", json={"question": "What is the widget code?"}).json()
    assert not denied["grounded"]
    ok = client.post("/ask", json={"question": "What is the widget code?"}, headers={"X-Groups": "eng"}).json()
    assert ok["grounded"] and "4242" in ok["answer"]
    assert "x-request-id" in {k.lower() for k in client.get("/health").headers}
    a = client.post("/agent", json={"question": "What is the widget code?"}, headers={"X-Groups": "eng"}).json()
    assert a["tool_calls"] >= 1 and "4242" in a["answer"]
