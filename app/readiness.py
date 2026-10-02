"""Readiness checks that avoid loading any local AI model."""

from pathlib import Path

import httpx
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, text

from app.api.schemas import DependencyCheck, ReadinessResponse
from app.config import Settings


class ReadinessChecker:
    def __init__(self, *, settings: Settings, engine: Engine) -> None:
        self._settings = settings
        self._engine = engine

    def check(self) -> ReadinessResponse:
        checks = {
            "configuration": DependencyCheck(status="ok"),
            "database": self._check_database(),
            "migrations": self._check_migrations(),
            "ollama": self._check_ollama(),
        }
        status = (
            "ready"
            if all(item.status == "ok" for item in checks.values())
            else "not_ready"
        )
        return ReadinessResponse(status=status, checks=checks)

    def _check_database(self) -> DependencyCheck:
        try:
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return DependencyCheck(status="ok")
        except Exception:
            return DependencyCheck(status="error", detail="database is unreachable")

    def _check_migrations(self) -> DependencyCheck:
        try:
            root = Path(__file__).resolve().parent.parent
            config = Config(str(root / "alembic.ini"))
            head = ScriptDirectory.from_config(config).get_current_head()
            with self._engine.connect() as connection:
                current = connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one()
            if current != head:
                return DependencyCheck(
                    status="error", detail="database migrations are not at head"
                )
            return DependencyCheck(status="ok")
        except Exception:
            return DependencyCheck(
                status="error", detail="database migration state is unavailable"
            )

    def _check_ollama(self) -> DependencyCheck:
        if self._settings.llm_provider != "ollama":
            return DependencyCheck(status="ok")
        try:
            response = httpx.get(
                f"{self._settings.ollama_host.rstrip('/')}/api/tags",
                timeout=self._settings.readiness_timeout_seconds,
            )
            response.raise_for_status()
            model_names = {
                str(model.get("name", ""))
                for model in response.json().get("models", [])
            }
            if self._settings.llm_model not in model_names:
                return DependencyCheck(
                    status="error", detail="configured Ollama model is unavailable"
                )
            return DependencyCheck(status="ok")
        except Exception:
            return DependencyCheck(status="error", detail="Ollama is unreachable")
