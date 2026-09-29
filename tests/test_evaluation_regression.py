import pytest

from evals.regression import MetricPolicy, compare_reports, flatten_numeric_metrics


def _report(metrics):
    return {
        "evaluation_version": "m11-v1",
        "metrics": metrics,
        "latencies_seconds": {},
    }


def test_flatten_numeric_metrics_preserves_metric_paths() -> None:
    assert flatten_numeric_metrics(
        {"retrieval": {"vector": {"recall_at_5": 0.8}}}
    ) == {"retrieval.vector.recall_at_5": 0.8}


def test_retrieval_change_within_tolerance_is_unchanged() -> None:
    result = compare_reports(
        _report({"retrieval": {"recall_at_5": 0.8}}),
        _report({"retrieval": {"recall_at_5": 0.79}}),
    )
    assert result["overall"] == "unchanged"


def test_retrieval_drop_beyond_tolerance_regresses() -> None:
    result = compare_reports(
        _report({"retrieval": {"recall_at_5": 0.8}}),
        _report({"retrieval": {"recall_at_5": 0.75}}),
    )
    assert result["overall"] == "regressed"


def test_deterministic_metric_requires_exact_baseline() -> None:
    result = compare_reports(
        _report({"deterministic": {"financial_accuracy": 1.0}}),
        _report({"deterministic": {"financial_accuracy": 0.99}}),
    )
    assert result["overall"] == "regressed"


def test_improvement_is_reported_per_metric() -> None:
    result = compare_reports(
        _report({"citation": {"support_rate": 0.8}}),
        _report({"citation": {"support_rate": 0.9}}),
    )
    assert result["overall"] == "improved"
    assert result["metrics"][0]["status"] == "improved"


def test_missing_metric_is_a_regression() -> None:
    result = compare_reports(
        _report({"retrieval": {"recall_at_5": 0.8}}),
        _report({"retrieval": {}}),
    )
    assert result["overall"] == "regressed"
    assert result["missing_metrics"] == ["retrieval.recall_at_5"]


def test_policy_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        MetricPolicy("higher", -0.1)


def test_better_refusal_behavior_is_an_improvement() -> None:
    result = compare_reports(
        _report({"refusal": {"correct_behavior_rate": 0.6, "case_count": 5}}),
        _report({"refusal": {"correct_behavior_rate": 0.8, "case_count": 5}}),
    )
    by_metric = {item["metric"]: item["status"] for item in result["metrics"]}
    assert by_metric["refusal.correct_behavior_rate"] == "improved"
    assert by_metric["refusal.case_count"] == "informational"
    assert result["overall"] == "improved"


def test_latency_changes_are_informational() -> None:
    baseline = _report({"retrieval": {"recall_at_5": 0.8}})
    current = _report({"retrieval": {"recall_at_5": 0.8}})
    baseline["latencies_seconds"] = {"overall": 10.0}
    current["latencies_seconds"] = {"overall": 12.0}
    result = compare_reports(baseline, current)
    by_metric = {item["metric"]: item["status"] for item in result["metrics"]}
    assert by_metric["latency.overall"] == "informational"
    assert result["overall"] == "unchanged"
