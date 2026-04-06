"""Hypotheses route coverage."""

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


def _create_ready_document(
    client: TestClient,
    headers: dict[str, str],
    workspace_id: str,
    filename: str,
    body: str,
) -> str:
    upload = client.post(
        f"/workspaces/{workspace_id}/documents/upload-url",
        headers=headers,
        json={
            "filename": filename,
            "file_format": "text",
            "file_size_bytes": len(body),
        },
    )
    assert upload.status_code == 201
    upload_body: dict[str, Any] = upload.json()

    parsed = urlparse(upload_body["upload_url"])
    upload_path = parsed.path if not parsed.query else f"{parsed.path}?{parsed.query}"
    put_result = client.put(
        upload_path,
        content=body.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
    )
    assert put_result.status_code == 204

    job_id = upload_body["job_id"]
    final_status = None
    for _ in range(25):
        poll = client.get(f"/jobs/{job_id}", headers=headers)
        assert poll.status_code == 200
        final_status = poll.json()["status"]
        if final_status in {"ready", "failed"}:
            break
        time.sleep(0.05)

    assert final_status == "ready"
    return str(upload_body["document"]["id"])


def _build_graph(client: TestClient, headers: dict[str, str], workspace_id: str, document_ids: list[str]) -> str:
    build = client.post(
        f"/workspaces/{workspace_id}/graphs",
        headers=headers,
        json={
            "document_ids": document_ids,
            "strategy": "domain_network",
            "plan_strategy": "weak-evidence",
            "max_plan_actions": 10,
        },
    )
    assert build.status_code == 201
    body: dict[str, Any] = build.json()

    graph_job_id = body["job_id"]
    final_status = None
    for _ in range(25):
        poll = client.get(f"/jobs/{graph_job_id}", headers=headers)
        assert poll.status_code == 200
        final_status = poll.json()["status"]
        if final_status in {"ready", "failed"}:
            break
        time.sleep(0.05)

    assert final_status == "ready"
    return str(body["graph"]["id"])


def test_get_hypotheses_returns_empty_when_no_bridge_hypothesis(client: TestClient) -> None:
    headers = _auth_headers(client, "hyp-empty@example.com")

    workspace = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Hypothesis Empty", "slug": "hypothesis-empty"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    doc1 = _create_ready_document(client, headers, workspace_id, "a.txt", "Linear pipeline for docs A")
    doc2 = _create_ready_document(client, headers, workspace_id, "b.txt", "Sequential pipeline for docs B")
    graph_id = _build_graph(client, headers, workspace_id, [doc1, doc2])

    response = client.get(
        f"/workspaces/{workspace_id}/graphs/{graph_id}/hypotheses",
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json() == []


def test_get_hypotheses_after_task_runs(client: TestClient) -> None:
    headers = _auth_headers(client, "hyp-bridge@example.com")

    workspace = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Hypothesis Bridge", "slug": "hypothesis-bridge"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    doc1 = _create_ready_document(
        client,
        headers,
        workspace_id,
        "ml-cyber-1.txt",
        "Machine learning model detects intrusion and anomaly patterns for cybersecurity systems.",
    )
    doc2 = _create_ready_document(
        client,
        headers,
        workspace_id,
        "ml-cyber-2.txt",
        "Neural training improves threat detection and security anomaly classification.",
    )
    graph_id = _build_graph(client, headers, workspace_id, [doc1, doc2])

    response = client.get(
        f"/workspaces/{workspace_id}/graphs/{graph_id}/hypotheses",
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) >= 1
    assert body[0]["status"] == "proposed"
    assert isinstance(body[0]["hypothesis_text"], str)


def test_patch_hypothesis_updates_status(client: TestClient) -> None:
    headers = _auth_headers(client, "hyp-update@example.com")

    workspace = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Hypothesis Update", "slug": "hypothesis-update"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    doc1 = _create_ready_document(
        client,
        headers,
        workspace_id,
        "bridge-1.txt",
        "Federated learning enables privacy preserving analytics over distributed intrusion sensors.",
    )
    doc2 = _create_ready_document(
        client,
        headers,
        workspace_id,
        "bridge-2.txt",
        "Cybersecurity anomaly scoring improves with distributed inference and privacy controls.",
    )
    graph_id = _build_graph(client, headers, workspace_id, [doc1, doc2])

    listed = client.get(f"/workspaces/{workspace_id}/graphs/{graph_id}/hypotheses", headers=headers)
    assert listed.status_code == 200
    items = listed.json()
    assert items

    hypothesis_id = items[0]["id"]
    patched = client.patch(
        f"/workspaces/{workspace_id}/graphs/{graph_id}/hypotheses/{hypothesis_id}",
        headers=headers,
        json={"status": "investigating"},
    )
    assert patched.status_code == 200
    assert patched.json()["status"] == "investigating"


def test_patch_hypothesis_by_non_member_returns_403(client: TestClient) -> None:
    owner_headers = _auth_headers(client, "owner-hyp@example.com")
    outsider_headers = _auth_headers(client, "outsider-hyp@example.com")

    workspace = client.post(
        "/workspaces",
        headers=owner_headers,
        json={"name": "Protected Hypothesis", "slug": "protected-hypothesis"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    doc1 = _create_ready_document(
        client,
        owner_headers,
        workspace_id,
        "secure-1.txt",
        "Machine learning and cybersecurity overlap in anomaly modeling.",
    )
    doc2 = _create_ready_document(
        client,
        owner_headers,
        workspace_id,
        "secure-2.txt",
        "Distributed privacy techniques improve cyber defense telemetry.",
    )
    graph_id = _build_graph(client, owner_headers, workspace_id, [doc1, doc2])

    listed = client.get(f"/workspaces/{workspace_id}/graphs/{graph_id}/hypotheses", headers=owner_headers)
    assert listed.status_code == 200
    hypothesis_id = listed.json()[0]["id"]

    forbidden = client.patch(
        f"/workspaces/{workspace_id}/graphs/{graph_id}/hypotheses/{hypothesis_id}",
        headers=outsider_headers,
        json={"status": "rejected"},
    )
    assert forbidden.status_code == 403
