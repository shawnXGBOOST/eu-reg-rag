from fastapi.testclient import TestClient

from eu_reg_rag.api import app


def test_health_endpoint_responds():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["api"] == "ok"


def test_ask_endpoint_uses_pipeline_and_validates_input():
    from eu_reg_rag import api

    class FakePipeline:
        def ask(self, q):
            return {"question": q, "answer": "ok [GDPR Art. 6]", "supported": True}

    api.state["pipeline"] = FakePipeline()
    try:
        client = TestClient(app)
        r = client.post("/ask", json={"question": "What is lawful basis?"})
        assert r.status_code == 200 and r.json()["supported"] is True
        assert client.post("/ask", json={"question": "x"}).status_code == 422  # too short
    finally:
        api.state.clear()
    assert TestClient(app).post("/ask", json={"question": "anything"}).status_code == 503


def test_ask_stream_endpoint_returns_ndjson():
    import json

    from eu_reg_rag import api

    class FakeAgent:
        def ask_stream(self, q):
            yield {"type": "token", "text": "Hi"}
            yield {"type": "final", "answer": "Hi", "supported": True}

    api.state["pipeline"] = FakeAgent()
    try:
        r = TestClient(app).post("/ask/stream", json={"question": "hello there"})
        lines = [json.loads(x) for x in r.text.strip().splitlines()]
        assert [e["type"] for e in lines] == ["token", "final"]
    finally:
        api.state.clear()
