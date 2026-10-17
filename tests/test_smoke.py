from fastapi.testclient import TestClient

from eu_reg_rag.api import app


def test_health_endpoint_responds():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["api"] == "ok"
