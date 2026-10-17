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
