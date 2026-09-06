"""Deterministic tests for source-backed revenue growth."""

from decimal import Decimal

import pytest

from app.financial.revenue_growth import (
    AmbiguousFinancialInputError,
    InvalidFinancialInputError,
    MissingFinancialInputError,
    SourceFinancialValue,
    calculate_revenue_growth,
    extract_revenue_growth_inputs,
)
from app.retrieval.vector_store import VectorSearchResult


def _value(year: int, amount: str, *, unit: str = "USD millions") -> SourceFinancialValue:
    return SourceFinancialValue(
        metric="total_net_sales",
        fiscal_year=year,
        amount=Decimal(amount),
        unit=unit,
        evidence_id=1,
        chunk_id=141,
        page_number=26,
        document_id=1,
    )


def _evidence(text: str, *, chunk_id: int = 141, page: int = 26) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        page_number=page,
        document_id=1,
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        cosine_distance=0.1,
        similarity=0.9,
    )


def test_positive_revenue_growth() -> None:
    result = calculate_revenue_growth(_value(2025, "110"), _value(2024, "100"))
    assert result.percentage_change == Decimal("10.0")


def test_revenue_decline() -> None:
    result = calculate_revenue_growth(_value(2025, "90"), _value(2024, "100"))
    assert result.percentage_change == Decimal("-10.0")


@pytest.mark.parametrize("prior_amount", ["0", "-1"])
def test_zero_or_invalid_denominator(prior_amount: str) -> None:
    if prior_amount == "-1":
        with pytest.raises(InvalidFinancialInputError, match="must not be negative"):
            _value(2024, prior_amount)
        return
    with pytest.raises(InvalidFinancialInputError, match="greater than zero"):
        calculate_revenue_growth(_value(2025, "100"), _value(2024, prior_amount))


@pytest.mark.parametrize(
    ("current", "prior", "message"),
    [(None, _value(2024, "100"), "current revenue"), (_value(2025, "100"), None, "prior revenue")],
)
def test_missing_inputs(current: SourceFinancialValue | None, prior: SourceFinancialValue | None, message: str) -> None:
    with pytest.raises(MissingFinancialInputError, match=message):
        calculate_revenue_growth(current, prior)


def test_rounding_uses_decimal_half_up() -> None:
    result = calculate_revenue_growth(
        _value(2025, "100.05"),
        _value(2024, "100"),
        rounding_places=1,
    )
    assert result.percentage_change == Decimal("0.1")


def test_extracts_change_column_table_and_preserves_provenance() -> None:
    result = _evidence(
        "The following table shows net sales for 2025 Change 2024 Change 2023 "
        "(dollars in millions): Total net sales $ 416,161 6 %$ 391,035 2 %$ 383,285"
    )
    current, prior = extract_revenue_growth_inputs(
        [result], current_year=2025, prior_year=2024
    )
    assert (current.amount, prior.amount) == (Decimal("416161"), Decimal("391035"))
    assert current.evidence_id == 1
    assert current.chunk_id == 141
    assert current.page_number == 26


def test_agreeing_duplicate_rows_are_not_ambiguous() -> None:
    row = "2025 2024 2023 (in millions): Total net sales 416,161 391,035 383,285"
    current, prior = extract_revenue_growth_inputs(
        [_evidence(row), _evidence(row, chunk_id=164, page=32)],
        current_year=2025,
        prior_year=2024,
    )
    assert current.evidence_id == 1
    assert prior.amount == Decimal("391035")


def test_conflicting_rows_are_rejected_as_ambiguous() -> None:
    with pytest.raises(AmbiguousFinancialInputError, match="conflicting"):
        extract_revenue_growth_inputs(
            [
                _evidence("2025 2024 (in millions): Total net sales 110 100"),
                _evidence(
                    "2025 2024 (in millions): Total net sales 111 100",
                    chunk_id=2,
                ),
            ],
            current_year=2025,
            prior_year=2024,
        )


def test_missing_explicit_row_is_rejected() -> None:
    with pytest.raises(MissingFinancialInputError, match="no explicit"):
        extract_revenue_growth_inputs(
            [_evidence("Revenue discussion for 2025 and 2024 (in millions).")],
            current_year=2025,
            prior_year=2024,
        )
