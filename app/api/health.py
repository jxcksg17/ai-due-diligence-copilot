"""
Health-check endpoints.

Two separate endpoints, deliberately not merged into one, because they
answer two different questions:

  GET /health     -> "Is the FastAPI process itself up?"
                      Must never depend on the database. If this ever
                      fails, the whole service is down.

  GET /health/db  -> "Can the app currently reach Postgres?"
                      This *can* fail independently (DB down, bad
                      credentials, network issue) without meaning the
                      API process itself is broken. Conflating the two
                      would make your test suite (and your production
                      alerting) flaky whenever the DB is briefly
                      unreachable for reasons unrelated to app code.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_env,
    }


@router.get("/health/db")
def health_db(db: Session = Depends(get_db)) -> dict:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "reachable"}
    except Exception as exc:  # noqa: BLE001 - intentionally broad for a health check
        return {"status": "error", "database": "unreachable", "detail": str(exc)}
