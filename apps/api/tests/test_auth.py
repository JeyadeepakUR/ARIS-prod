"""Auth endpoint tests for Sprint 1."""

from fastapi.testclient import TestClient


def test_register_endpoint(client: TestClient) -> None:
    response = client.post(
        "/auth/register",
        json={"email": "user@example.com", "password": "safepassword123"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str)
    assert isinstance(body["refresh_token"], str)
    assert body["user"]["email"] == "user@example.com"


def test_login_endpoint(client: TestClient) -> None:
    register_response = client.post(
        "/auth/register",
        json={"email": "login@example.com", "password": "safepassword123"},
    )
    assert register_response.status_code == 201

    response = client.post(
        "/auth/login",
        json={"email": "login@example.com", "password": "safepassword123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str)
    assert isinstance(body["refresh_token"], str)


def test_refresh_endpoint(client: TestClient) -> None:
    register_response = client.post(
        "/auth/register",
        json={"email": "refresh@example.com", "password": "safepassword123"},
    )
    assert register_response.status_code == 201

    refresh_token = register_response.json()["refresh_token"]
    response = client.post("/auth/refresh", json={"refresh_token": refresh_token})

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert isinstance(body["access_token"], str)
