from decimal import Decimal

import pytest

from evals.metrics import (
    anchor_coverage,
    classification_metrics,
    numeric_matches,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    refusal_accuracy,
)


def test_recall_at_k_counts_unique_relevant_targets() -> None:
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, 3) == 1.0


def test_recall_at_k_exposes_missed_target() -> None:
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 1) == 0.5


def test_precision_at_k_uses_requested_cutoff() -> None:
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)


def test_mrr_uses_first_relevant_rank() -> None:
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5


def test_retrieval_metrics_reject_empty_ground_truth() -> None:
    with pytest.raises(ValueError, match="at least one relevant"):
        recall_at_k(["a"], set(), 1)


def test_classification_metrics_are_macro_averaged() -> None:
    result = classification_metrics(
        ["supported", "supported", "ambiguous"],
        ["supported", "ambiguous", "ambiguous"],
    )
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.macro_precision == pytest.approx(0.75)
    assert result.macro_recall == pytest.approx(0.75)
    assert result.macro_f1 == pytest.approx(2 / 3)


def test_numeric_tolerance_is_decimal_safe() -> None:
    assert numeric_matches(Decimal("6.4001"), Decimal("6.4"), tolerance="0.001")
    assert not numeric_matches(Decimal("6.41"), Decimal("6.4"), tolerance="0.001")


def test_numeric_tolerance_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        numeric_matches(1, 1, tolerance=-1)


def test_refusal_accuracy_distinguishes_safe_refusal_from_answering() -> None:
    assert refusal_accuracy([True, True, False], [True, False, False]) == pytest.approx(
        2 / 3
    )


def test_answer_anchor_coverage_is_case_and_whitespace_insensitive() -> None:
    answer = "Apple reported  $416,161 million in total net sales."
    assert anchor_coverage(answer, ["416,161 MILLION", "total net sales"]) == 1.0


def test_answer_anchor_coverage_reports_partial_relevancy() -> None:
    assert anchor_coverage("Revenue was $10.", ["revenue", "profit"]) == 0.5
