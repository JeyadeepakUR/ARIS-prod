"""Sprint 3 integration test for ARIS graph pipeline endpoints."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse
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


def test_graph_build_with_three_docs_returns_edges_with_evidence(client: TestClient) -> None:
    headers = _auth_headers(client, "sprint3-graph@example.com")

    workspace = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Sprint Three", "slug": "sprint-three"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    document_ids: list[str] = []
    for index in range(3):
        upload = client.post(
            f"/workspaces/{workspace_id}/documents/upload-url",
            headers=headers,
            json={
                "filename": f"doc-{index + 1}.txt",
                "file_format": "text",
                "file_size_bytes": 128,
            },
        )
        assert upload.status_code == 201
        upload_body: dict[str, Any] = upload.json()
        document_ids.append(upload_body["document"]["id"])

        parsed = urlparse(upload_body["upload_url"])
        upload_path = parsed.path if not parsed.query else f"{parsed.path}?{parsed.query}"
        upload_put = client.put(
            upload_path,
            data=f"Document {index + 1} real content for graph build".encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
        assert upload_put.status_code == 204

        job_id = upload_body["job_id"]

        final_status = None
        for _ in range(25):
            poll = client.get(f"/jobs/{job_id}", headers=headers)
            assert poll.status_code == 200
            final_status = poll.json()["status"]
            if final_status == "ready":
                break
            time.sleep(0.05)
        assert final_status == "ready"

    list_docs = client.get(f"/workspaces/{workspace_id}/documents", headers=headers)
    assert list_docs.status_code == 200
    listed_documents: list[dict[str, Any]] = list_docs.json()
    assert len(listed_documents) == 3

    build = client.post(
        f"/workspaces/{workspace_id}/graphs",
        headers=headers,
        json={
            "document_ids": document_ids,
            "strategy": "sequential",
            "plan_strategy": "weak-evidence",
            "max_plan_actions": 10,
        },
    )
    assert build.status_code == 201
    body: dict[str, Any] = build.json()
    graph_id = body["graph"]["id"]
    graph_job_id = body["job_id"]

    graph_final_status = None
    for _ in range(25):
        poll = client.get(f"/jobs/{graph_job_id}", headers=headers)
        assert poll.status_code == 200
        graph_final_status = poll.json()["status"]
        if graph_final_status in {"ready", "failed"}:
            break
        time.sleep(0.05)

    assert graph_final_status == "ready"

    list_graphs = client.get(f"/workspaces/{workspace_id}/graphs", headers=headers)
    assert list_graphs.status_code == 200
    listed_graphs: list[dict[str, Any]] = list_graphs.json()
    assert len(listed_graphs) >= 1
    assert listed_graphs[0]["id"] == graph_id

    nodes_response = client.get(f"/graphs/{graph_id}/nodes", headers=headers)
    assert nodes_response.status_code == 200
    nodes: list[dict[str, Any]] = nodes_response.json()
    assert len(nodes) == 3
    assert isinstance(nodes[0]["label"], str)

    edges_response = client.get(f"/graphs/{graph_id}/edges", headers=headers)
    assert edges_response.status_code == 200
    edges: list[dict[str, Any]] = edges_response.json()
    assert len(edges) >= 2
    assert isinstance(edges[0]["evidence"], dict)
    assert isinstance(edges[0]["evidence"].get("text"), str)
    assert edges[0]["evidence"].get("text")


def test_domain_network_build_creates_domain_and_bridge_nodes(client: TestClient) -> None:
    headers = _auth_headers(client, "sprint3-domain-network@example.com")

    workspace = client.post(
        "/workspaces",
        headers=headers,
        json={"name": "Domain Network", "slug": "domain-network"},
    )
    assert workspace.status_code == 201
    workspace_id = workspace.json()["id"]

    docs_payload = [
        {
            "filename": "ml-cyber-paper-1.txt",
            "body": "Machine learning model detects intrusion and anomaly patterns for cybersecurity.",
        },
        {
            "filename": "ml-cyber-paper-2.txt",
            "body": "Neural training improves threat detection and security anomaly classification.",
        },
    ]

    document_ids: list[str] = []
    for payload in docs_payload:
        upload = client.post(
            f"/workspaces/{workspace_id}/documents/upload-url",
            headers=headers,
            json={
                "filename": payload["filename"],
                "file_format": "text",
                "file_size_bytes": len(payload["body"]),
            },
        )
        assert upload.status_code == 201
        upload_body: dict[str, Any] = upload.json()
        document_ids.append(upload_body["document"]["id"])

        parsed = urlparse(upload_body["upload_url"])
        upload_path = parsed.path if not parsed.query else f"{parsed.path}?{parsed.query}"
        put_result = client.put(
            upload_path,
            content=payload["body"].encode("utf-8"),
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
    graph_id = body["graph"]["id"]
    graph_job_id = body["job_id"]

    graph_final_status = None
    for _ in range(25):
        poll = client.get(f"/jobs/{graph_job_id}", headers=headers)
        assert poll.status_code == 200
        graph_final_status = poll.json()["status"]
        if graph_final_status in {"ready", "failed"}:
            break
        time.sleep(0.05)

    assert graph_final_status == "ready"

    nodes_response = client.get(f"/graphs/{graph_id}/nodes", headers=headers)
    assert nodes_response.status_code == 200
    nodes: list[dict[str, Any]] = nodes_response.json()
    node_types = {node["node_type"] for node in nodes}

    assert "document" in node_types
    assert "domain" in node_types
    assert "concept" in node_types
    assert "bridge_concept" in node_types

    edges_response = client.get(f"/graphs/{graph_id}/edges", headers=headers)
    assert edges_response.status_code == 200
    edges: list[dict[str, Any]] = edges_response.json()
    edge_types = {edge["edge_type"] for edge in edges}

    assert "belongs_to_domain" in edge_types
    assert "has_concept" in edge_types
    assert "cross_domain_bridge" in edge_types
