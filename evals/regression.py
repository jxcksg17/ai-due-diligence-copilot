"""Metric-by-metric comparison against an approved baseline."""

from dataclasses import asdict, dataclass
from typing import Any, Literal


Direction = Literal["higher", "lower", "exact", "informational"]


@dataclass(frozen=True)
class MetricPolicy:
    direction: Direction
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("regression tolerance must not be negative")


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    baseline: float
    current: float
    delta: float
    status: str
    tolerance: float
    direction: Direction


DEFAULT_POLICIES = {
    "retrieval": MetricPolicy("higher", 0.02),
    "generation": MetricPolicy("higher", 0.05),
    "citation": MetricPolicy("higher", 0.05),
    "deterministic": MetricPolicy("exact", 0.0),
    "classification": MetricPolicy("higher", 0.0),
    "refusal": MetricPolicy("higher", 0.0),
    "latency": MetricPolicy("informational", 0.0),
}


def flatten_numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_numeric_metrics(child, path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def compare_reports(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    policies: dict[str, MetricPolicy] | None = None,
) -> dict[str, Any]:
    active_policies = policies or DEFAULT_POLICIES
    old = flatten_numeric_metrics(baseline.get("metrics", {}))
    new = flatten_numeric_metrics(current.get("metrics", {}))
    old.update(
        {
            f"latency.{key}": value
            for key, value in flatten_numeric_metrics(
                baseline.get("latencies_seconds", {})
            ).items()
        }
    )
    new.update(
        {
            f"latency.{key}": value
            for key, value in flatten_numeric_metrics(
                current.get("latencies_seconds", {})
            ).items()
        }
    )
    comparisons: list[MetricComparison] = []
    missing = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    for metric in sorted(set(old) & set(new)):
        category = metric.split(".", 1)[0]
        policy = (
            MetricPolicy("informational", 0.0)
            if metric.endswith("count") or metric.endswith("_count")
            else active_policies.get(category, MetricPolicy("higher", 0.0))
        )
        comparisons.append(
            _compare_metric(metric, old[metric], new[metric], policy)
        )
    statuses = [item.status for item in comparisons]
    overall = (
        "regressed"
        if missing or "regressed" in statuses
        else "improved"
        if "improved" in statuses
        else "unchanged"
    )
    return {
        "baseline_evaluation_version": baseline.get("evaluation_version"),
        "current_evaluation_version": current.get("evaluation_version"),
        "overall": overall,
        "metrics": [asdict(item) for item in comparisons],
        "missing_metrics": missing,
        "added_metrics": added,
    }


def _compare_metric(
    name: str, baseline: float, current: float, policy: MetricPolicy
) -> MetricComparison:
    delta = current - baseline
    if policy.direction == "informational":
        status = "informational"
    elif policy.direction == "exact":
        status = "unchanged" if abs(delta) <= policy.tolerance else "regressed"
    elif policy.direction == "higher":
        status = (
            "regressed"
            if delta < -policy.tolerance
            else "improved"
            if delta > policy.tolerance
            else "unchanged"
        )
    else:
        status = (
            "regressed"
            if delta > policy.tolerance
            else "improved"
            if delta < -policy.tolerance
            else "unchanged"
        )
    return MetricComparison(
        metric=name,
        baseline=baseline,
        current=current,
        delta=delta,
        status=status,
        tolerance=policy.tolerance,
        direction=policy.direction,
    )
