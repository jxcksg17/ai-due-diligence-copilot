"""
Shared pytest fixtures.

Critically, this sets a test-safe DATABASE_URL in the environment
*before* anything imports `app.main` / `app.config`. `Settings.database_url`
is a required field with no default, so importing the app without a
DATABASE_URL set anywhere would fail immediately at collection time.

Note this test suite does not require a real, running Postgres server:
`GET /health` never touches the database, and the one test we add for
M1 only exercises that endpoint. `GET /health/db` will be exercised by
an integration test once we actually have a database to run
integration tests against (later milestone) — testing it against a
fake/unreachable URL right now would only prove the error branch,
which isn't meaningful yet.
"""

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test_db"
)

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
