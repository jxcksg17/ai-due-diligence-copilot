"""Small, transparent metrics used by the M11 evaluator."""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from math import isclose
from typing import Hashable, Iterable, Sequence


def recall_at_k(
    ranked: Sequence[Hashable], relevant: set[Hashable], k: int
) -> float:
    _validate_k(k)
    if not relevant:
        raise ValueError("recall requires at least one relevant target")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(
    ranked: Sequence[Hashable], relevant: set[Hashable], k: int
) -> float:
    _validate_k(k)
    if not relevant:
        raise ValueError("precision requires at least one relevant target")
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: Sequence[Hashable], relevant: set[Hashable]) -> float:
    if not relevant:
        raise ValueError("reciprocal rank requires at least one relevant target")
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("mean requires at least one value")
    return sum(materialized) / len(materialized)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    count: int


def classification_metrics(
    expected: Sequence[str], predicted: Sequence[str]
) -> ClassificationMetrics:
    if not expected or len(expected) != len(predicted):
        raise ValueError("classification labels must be non-empty and aligned")
    labels = sorted(set(expected) | set(predicted))
    pairs = list(zip(expected, predicted, strict=True))
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(e == label and p == label for e, p in pairs)
        false_positive = sum(e != label and p == label for e, p in pairs)
        false_negative = sum(e == label and p != label for e, p in pairs)
        precision = _safe_ratio(true_positive, true_positive + false_positive)
        recall = _safe_ratio(true_positive, true_positive + false_negative)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return ClassificationMetrics(
        accuracy=sum(e == p for e, p in pairs) / len(pairs),
        macro_precision=mean(precision_values),
        macro_recall=mean(recall_values),
        macro_f1=mean(f1_values),
        count=len(pairs),
    )


def numeric_matches(
    actual: Decimal | float,
    expected: Decimal | float,
    *,
    tolerance: Decimal | float = 0,
) -> bool:
    actual_decimal = Decimal(str(actual))
    expected_decimal = Decimal(str(expected))
    tolerance_decimal = Decimal(str(tolerance))
    if tolerance_decimal < 0:
        raise ValueError("numeric tolerance must not be negative")
    return abs(actual_decimal - expected_decimal) <= tolerance_decimal


def refusal_accuracy(expected: Sequence[bool], predicted: Sequence[bool]) -> float:
    if not expected or len(expected) != len(predicted):
        raise ValueError("refusal labels must be non-empty and aligned")
    return sum(e is p for e, p in zip(expected, predicted, strict=True)) / len(expected)


def anchor_coverage(answer: str, expected_anchors: Sequence[str]) -> float:
    """Fraction of reviewed fact anchors found after whitespace/case normalization."""
    if not expected_anchors:
        raise ValueError("answer relevancy requires at least one expected anchor")
    normalized = _normalize(answer)
    matched = sum(_normalize(anchor) in normalized for anchor in expected_anchors)
    return matched / len(expected_anchors)


def label_counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
