from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from app.api.concurrency import AIConcurrencyGuard
from app.api.errors import AIConcurrencyLimitError, ModelUnavailableError
from app.api.schemas import (
    DependencyCheck,
    QueryResponse,
    ReadinessResponse,
)
from app.config import Settings
from app.db.session import get_db
from app.main import create_app
from app.readiness import ReadinessChecker


class FakeReadiness:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready

    def check(self) -> ReadinessResponse:
        state = "ok" if self.ready else "error"
        return ReadinessResponse(
            status="ready" if self.ready else "not_ready",
            checks={"database": DependencyCheck(status=state)},
        )


class FakeService:
    def query(self, db, request):
        return QueryResponse(
            answer="Apple reported the requested fact [1].",
            citation_ids=[1],
            insufficient_evidence=False,
            verification_state="verified",
            evidence=[],
            claim_verifications=[],
        )

    def compare_revenue(self, db, request):
        raise ModelUnavailableError("local model provider is unavailable")

    def compare_risks(self, db, request):  # pragma: no cover - protocol fixture
        raise NotImplementedError

    def assess_claim(self, db, request):  # pragma: no cover - protocol fixture
        raise NotImplementedError


@pytest.fixture
def production_client() -> TestClient:
    settings = Settings(
        database_url="postgresql+psycopg://test:test@localhost/test",
        app_env="test",
        debug=False,
        max_request_body_bytes=300,
    )
    app = create_app(
        settings=settings,
        production_api=FakeService(),
        readiness_checker=FakeReadiness(),
    )

    def fake_db():
        yield object()

    app.dependency_overrides[get_db] = fake_db
    return TestClient(app, raise_server_exceptions=False)


def test_liveness_and_readiness_are_distinct(production_client: TestClient) -> None:
    live = production_client.get("/health/live")
    ready = production_client.get("/health/ready")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert live.headers["x-request-id"]


def test_not_ready_returns_503() -> None:
    settings = Settings(
        database_url="postgresql+psycopg://test:test@localhost/test",
        app_env="test",
        debug=False,
    )
    app = create_app(settings=settings, production_api=FakeService(), readiness_checker=FakeReadiness(False))
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_grounded_query_contract(production_client: TestClient) -> None:
    response = production_client.post(
        "/api/v1/query",
        json={"question": "What were net sales?", "company": "Apple"},
        headers={"X-Request-ID": "portfolio-smoke-1"},
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "portfolio-smoke-1"
    assert response.json()["verification_state"] == "verified"


def test_validation_uses_safe_error_envelope(production_client: TestClient) -> None:
    response = production_client.post(
        "/api/v1/query", json={"question": "x", "company": "Apple", "extra": True}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["request_id"]


def test_model_unavailable_is_503(production_client: TestClient) -> None:
    response = production_client.post(
        "/api/v1/temporal/revenue",
        json={"company": "Apple", "older_year": 2024, "newer_year": 2025},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    assert response.json()["error"]["request_id"] != "unknown"


def test_request_size_limit(production_client: TestClient) -> None:
    response = production_client.post(
        "/api/v1/query",
        json={"question": "q" * 400, "company": "Apple"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_openapi_exposes_only_versioned_business_routes(production_client: TestClient) -> None:
    paths = production_client.get("/openapi.json").json()["paths"]
    assert "/api/v1/query" in paths
    assert "/api/v1/temporal/revenue" in paths
    assert "/api/v1/risk-radar/compare" in paths
    assert "/api/v1/claim-evidence/assess" in paths
    assert "/health/live" in paths
    assert "/health/ready" in paths


def test_production_configuration_rejects_debug_and_non_postgres() -> None:
    with pytest.raises(ValueError, match="DEBUG must be false"):
        Settings(database_url="sqlite:///bad.db", app_env="production", debug=True)
    with pytest.raises(ValueError, match=r"postgresql\+psycopg"):
        Settings(database_url="sqlite:///bad.db", app_env="production", debug=False)


def test_cors_is_explicit_and_exposes_request_id(production_client: TestClient) -> None:
    response = production_client.options(
        "/api/v1/query",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-request-id",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    actual = production_client.post(
        "/api/v1/query",
        json={"question": "What were net sales?", "company": "Apple"},
        headers={"Origin": "http://localhost:5173"},
    )
    assert "X-Request-ID" in actual.headers["access-control-expose-headers"]


def test_cors_rejects_wildcard_origin_configuration() -> None:
    with pytest.raises(ValueError, match="explicit origins"):
        Settings(
            database_url="postgresql+psycopg://test:test@localhost/test",
            cors_allowed_origins=["*"],
        )


def test_concurrency_guard_rejects_second_caller() -> None:
    guard = AIConcurrencyGuard(limit=1, queue_timeout_seconds=0.01)
    entered = Event()
    release = Event()

    def occupy() -> None:
        with guard.slot():
            entered.set()
            release.wait(timeout=1)

    worker = Thread(target=occupy)
    worker.start()
    assert entered.wait(timeout=1)
    with pytest.raises(AIConcurrencyLimitError):
        with guard.slot():
            pass
    release.set()
    worker.join(timeout=1)


class _ScalarResult:
    def __init__(self, value=None) -> None:
        self.value = value

    def scalar_one(self):
        return self.value


class _Connection:
    def __init__(self, revision: str) -> None:
        self.revision = revision

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def execute(self, statement):
        if "alembic_version" in str(statement):
            return _ScalarResult(self.revision)
        return _ScalarResult()


class _Engine:
    def __init__(self, revision: str) -> None:
        self.revision = revision

    def connect(self):
        return _Connection(self.revision)


class _TagsResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"name": "qwen3:8b-q4_K_M"}]}


def test_readiness_checks_migration_and_ollama_without_loading_models(monkeypatch) -> None:
    settings = Settings(
        database_url="postgresql+psycopg://test:test@localhost/test",
        app_env="test",
        debug=False,
    )
    monkeypatch.setattr("app.readiness.httpx.get", lambda *args, **kwargs: _TagsResponse())
    result = ReadinessChecker(
        settings=settings,
        engine=_Engine("20260906_03"),
    ).check()
    assert result.status == "ready"
    assert set(result.checks) == {"configuration", "database", "migrations", "ollama"}


def test_readiness_rejects_stale_migration(monkeypatch) -> None:
    settings = Settings(
        database_url="postgresql+psycopg://test:test@localhost/test",
        app_env="test",
        debug=False,
    )
    monkeypatch.setattr("app.readiness.httpx.get", lambda *args, **kwargs: _TagsResponse())
    result = ReadinessChecker(settings=settings, engine=_Engine("old_revision")).check()
    assert result.status == "not_ready"
    assert result.checks["migrations"].status == "error"
