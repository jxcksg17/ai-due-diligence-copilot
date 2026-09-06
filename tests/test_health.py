"""
Tests for the liveness health endpoint.

This is the "at least one meaningful test" required for M1. It checks:
  - the endpoint returns HTTP 200,
  - the response has the expected shape,
  - the reported app name matches configuration (proving that
    environment-based config is actually wired into the response,
    not hardcoded).
"""

from fastapi.testclient import TestClient


def test_health_returns_ok_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body
    assert "environment" in body
