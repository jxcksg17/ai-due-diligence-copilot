"""Compact, atomic evaluation-report serialization."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "1"


def build_report(
    *,
    evaluation_version: str,
    git_commit: str,
    mode: str,
    models: dict[str, str],
    case_counts: dict[str, int],
    metrics: dict[str, Any],
    latencies: dict[str, Any],
    runtime: dict[str, Any],
    case_outcomes: dict[str, str] | None = None,
) -> dict[str, Any]:
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_version": evaluation_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "mode": mode,
        "models": models,
        "case_counts": case_counts,
        "metrics": metrics,
        "latencies_seconds": latencies,
        "runtime": runtime,
        "mandatory_external_api_cost_usd": 0,
    }
    if case_outcomes is not None:
        report["case_outcomes"] = dict(sorted(case_outcomes.items()))
    return report


def write_report_atomic(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_report(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "report_schema_version",
        "evaluation_version",
        "git_commit",
        "metrics",
        "models",
        "case_counts",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"evaluation report is missing fields: {', '.join(sorted(missing))}")
    return report
