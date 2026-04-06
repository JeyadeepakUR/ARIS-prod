"""Sprint 2 tests for workspace/document/job flow."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from fastapi.testclient import TestClient


def _auth_headers(client: TestClient, email: str) -> dict[str, str]:
    register = client.post(
        "/auth/register",
        json={"email": email, "password": "StrongPass123"},
    )
    assert register.status_code == 201
    token = register.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_workspace_crud(client: TestClient) -> None:
    headers = _auth_headers(client, "sprint2-workspace@example.com")

    create = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Sprint Two", "slug": "sprint-two"},
    )
    assert create.status_code == 201
    workspace = create.json()
    workspace_id = workspace["id"]

    fetch = client.get(f"/workspaces/{workspace_id}", headers=headers)
    assert fetch.status_code == 200
    assert fetch.json()["slug"] == "sprint-two"

    listed = client.get("/workspaces", headers=headers)
    assert listed.status_code == 200
    assert len(listed.json()) == 1


def test_upload_pdf_pending_to_ready_with_job_polling(client: TestClient) -> None:
    headers = _auth_headers(client, "sprint2-upload@example.com")

    workspace_response = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Uploads", "slug": "uploads"},
    )
    assert workspace_response.status_code == 201
    workspace_id = workspace_response.json()["id"]

    upload = client.post(
        f"/workspaces/{workspace_id}/documents/upload-url",
        headers=headers,
        json={"filename": "paper.txt", "file_format": "text", "file_size_bytes": 42},
    )

    assert upload.status_code == 201
    body: dict[str, Any] = upload.json()
    assert body["document"]["status"] in {"pending", "processing", "ready"}
    assert body["upload_url"].startswith("http")

    upload_url = body["upload_url"]
    parsed = urlparse(upload_url)
    upload_path = parsed.path if not parsed.query else f"{parsed.path}?{parsed.query}"
    upload_put = client.put(
        upload_path,
        content=b"Real uploaded content for sprint2 ingest test",
        headers={"Content-Type": "text/plain"},
    )
    assert upload_put.status_code == 204

    job_id = body["job_id"]

    final_status = None
    for _ in range(25):
        poll = client.get(f"/jobs/{job_id}", headers=headers)
        assert poll.status_code == 200
        final_status = poll.json()["status"]
        if final_status == "ready":
            break
        time.sleep(0.05)

    assert final_status == "ready"
