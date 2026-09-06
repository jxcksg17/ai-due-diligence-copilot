"""Deterministic cross-filing numeric comparison tests."""

from decimal import Decimal

import pytest

from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.temporal.financial import (
    TemporalFinancialInputError,
    calculate_temporal_numeric_change,
    extract_period_total_net_sales,
)
from app.temporal.models import (
    SourcePeriod,
    TemporalChangeCategory,
    TemporalEvidence,
    TemporalFinancialValue,
)


def _result(chunk_id: int, document_id: int, year: int, text: str = "evidence") -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        page_number=30 + chunk_id,
        document_id=document_id,
        company="Apple",
        document_type="10-K",
        fiscal_year=year,
        cosine_distance=0.1,
        similarity=0.9,
    )


def _value(
    amount: str,
    *,
    year: int,
    document_id: int,
    evidence_id: int,
    period: SourcePeriod,
) -> TemporalFinancialValue:
    return TemporalFinancialValue(
        metric="total_net_sales",
        amount=Decimal(amount),
        unit="USD millions",
        fiscal_year=year,
        evidence=TemporalEvidence(
            evidence_id=evidence_id,
            source_period=period,
            result=_result(evidence_id, document_id, year),
        ),
    )


@pytest.mark.parametrize(
    ("older_amount", "newer_amount", "category", "absolute", "percentage"),
    [
        ("100", "125", TemporalChangeCategory.INCREASED, "25", "25.0"),
        ("100", "80", TemporalChangeCategory.DECLINED, "-20", "-20.0"),
        ("100", "100", TemporalChangeCategory.UNCHANGED, "0", "0.0"),
    ],
)
def test_numeric_increase_decline_and_unchanged(
    older_amount: str,
    newer_amount: str,
    category: TemporalChangeCategory,
    absolute: str,
    percentage: str,
) -> None:
    result = calculate_temporal_numeric_change(
        _value(older_amount, year=2024, document_id=2, evidence_id=1, period=SourcePeriod.OLDER),
        _value(newer_amount, year=2025, document_id=1, evidence_id=2, period=SourcePeriod.NEWER),
    )
    assert result.category == category
    assert result.absolute_change == Decimal(absolute)
    assert result.percentage_change == Decimal(percentage)


def test_zero_denominator_preserves_absolute_change_and_marks_percentage_undefined() -> None:
    result = calculate_temporal_numeric_change(
        _value("0", year=2024, document_id=2, evidence_id=1, period=SourcePeriod.OLDER),
        _value("5", year=2025, document_id=1, evidence_id=2, period=SourcePeriod.NEWER),
    )
    assert result.absolute_change == Decimal("5")
    assert result.percentage_change is None
    assert "undefined" in result.as_prompt_block()


def test_missing_value_is_explicit() -> None:
    with pytest.raises(TemporalFinancialInputError, match="both temporal values"):
        calculate_temporal_numeric_change(
            None,
            _value("5", year=2025, document_id=1, evidence_id=2, period=SourcePeriod.NEWER),
        )


def test_extracts_each_filings_own_value_with_temporal_provenance() -> None:
    document = DocumentMetadata(2, "Apple", "10-K", 2024)
    source = _result(
        10,
        2,
        2024,
        "CONSOLIDATED STATEMENTS OF OPERATIONS\n"
        "The table shows values for 2024 and 2023 (in millions):\n"
        "2024 2023\nTotal net sales $ 391,035 $ 383,285",
    )
    value = extract_period_total_net_sales(
        [source],
        document=document,
        source_period=SourcePeriod.OLDER,
        evidence_id=1,
    )
    assert value.amount == Decimal("391035")
    assert value.fiscal_year == 2024
    assert value.evidence.result.document_id == 2
    assert value.evidence.result.page_number == 40
    assert value.evidence.source_period == SourcePeriod.OLDER


def test_conflicting_financial_evidence_is_rejected() -> None:
    document = DocumentMetadata(2, "Apple", "10-K", 2024)
    common = (
        "CONSOLIDATED STATEMENTS OF OPERATIONS\n"
        "The table shows values for 2024 and 2023 (in millions):\n2024 2023\n"
    )
    evidence = [
        _result(10, 2, 2024, common + "Total net sales $ 100 $ 90"),
        _result(11, 2, 2024, common + "Total net sales $ 101 $ 90"),
    ]
    with pytest.raises(TemporalFinancialInputError, match="conflicting"):
        extract_period_total_net_sales(
            evidence,
            document=document,
            source_period=SourcePeriod.OLDER,
            evidence_id=1,
        )


def test_ignores_total_net_sales_from_a_different_table() -> None:
    document = DocumentMetadata(1, "Apple", "10-K", 2025)
    statement = _result(
        10,
        1,
        2025,
        "CONSOLIDA TED ST ATEMENTS OF OPERA TIONS\n"
        "The table shows values for 2025 and 2024 (in millions):\n"
        "2025 2024\nTotal net sales $ 416,161 $ 391,035",
    )
    segment_table = _result(
        11,
        1,
        2025,
        "Geographic segments (in millions)\n2025 2024\n"
        "Total net sales $ 178,353 $ 111,032",
    )

    value = extract_period_total_net_sales(
        [segment_table, statement],
        document=document,
        source_period=SourcePeriod.NEWER,
        evidence_id=2,
    )

    assert value.amount == Decimal("416161")
    assert value.evidence.result.chunk_id == 10


def test_rejects_total_net_sales_without_statement_identity() -> None:
    document = DocumentMetadata(1, "Apple", "10-K", 2025)
    segment_table = _result(
        11,
        1,
        2025,
        "Geographic segments (in millions)\n2025 2024\n"
        "Total net sales $ 178,353 $ 111,032",
    )

    with pytest.raises(TemporalFinancialInputError, match="Statements of Operations"):
        extract_period_total_net_sales(
            [segment_table],
            document=document,
            source_period=SourcePeriod.NEWER,
            evidence_id=2,
        )


def test_cross_filing_provenance_requires_distinct_documents() -> None:
    with pytest.raises(TemporalFinancialInputError, match="distinct documents"):
        calculate_temporal_numeric_change(
            _value("100", year=2024, document_id=1, evidence_id=1, period=SourcePeriod.OLDER),
            _value("110", year=2025, document_id=1, evidence_id=2, period=SourcePeriod.NEWER),
        )
