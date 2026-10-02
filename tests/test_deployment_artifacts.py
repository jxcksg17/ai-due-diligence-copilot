import importlib.util
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


ROOT = Path(__file__).resolve().parent.parent


def test_container_runs_as_non_root_with_one_worker() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER app" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert "COPY data" not in dockerfile


def test_container_context_excludes_sensitive_and_large_local_files() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for entry in (".env", ".venv", "data/*.pdf", ".cache", ".ollama", "*.gguf"):
        assert entry in ignored


def test_compose_uses_explicit_migration_gate_and_external_ollama() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert 'command: ["alembic", "upgrade", "head"]' in compose
    assert "condition: service_completed_successfully" in compose
    assert "host.docker.internal" in compose
    assert "POSTGRES_PASSWORD:?" in compose


def test_ci_runs_migrations_tests_and_model_free_regression_gate() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "alembic upgrade head" in workflow
    assert "alembic check" in workflow
    assert "pytest -q" in workflow
    assert "python -m evals.ci_gate" in workflow


def test_packaged_migration_head_matches_verified_database_revision() -> None:
    config = Config(str(ROOT / "alembic.ini"))
    scripts = ScriptDirectory.from_config(config)
    assert scripts.get_current_head() == "20260906_03"
    assert [revision.revision for revision in scripts.walk_revisions()] == [
        "20260906_03",
        "20260905_02",
        "20260905_01",
        "20260905_00",
    ]


def test_baseline_migration_contains_only_original_ingestion_tables(monkeypatch) -> None:
    migration_path = (
        ROOT / "alembic" / "versions" / "20260905_00_baseline_ingestion_schema.py"
    )
    spec = importlib.util.spec_from_file_location("baseline_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    created: dict[str, tuple] = {}
    monkeypatch.setattr(
        migration.op,
        "create_table",
        lambda name, *objects, **kwargs: created.setdefault(name, objects),
    )
    migration.upgrade()
    assert set(created) == {"companies", "documents", "chunks"}
    chunk_columns = {
        item.name
        for item in created["chunks"]
        if hasattr(item, "name") and item.name is not None
    }
    assert chunk_columns == {
        "id",
        "document_id",
        "chunk_index",
        "page_number",
        "text",
        "section",
    }
    assert "embedding" not in chunk_columns
