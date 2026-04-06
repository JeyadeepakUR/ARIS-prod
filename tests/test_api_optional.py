"""Optional API tests (run only when FastAPI is installed)."""

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from aris.api.app import create_app


def test_health_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "aris"


def test_analyze_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    payload = {
        "documents": [
            {
                "document_id": "a",
                "domain": "nlp",
                "text": "Method A improves translation quality and supports robust transfer.",
            },
            {
                "document_id": "b",
                "domain": "robotics",
                "text": "Method A fails transfer in robotics and contradicts robustness claims.",
            },
        ],
        "top_k_bridges": 10,
        "max_hypotheses": 10,
    }

    response = client.post("/v1/analyze", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["profile_count"] == 2
    assert body["hypothesis_count"] >= 1
