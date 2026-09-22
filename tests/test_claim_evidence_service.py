"""Authoritative claim-vs-evidence state and arithmetic tests."""

from dataclasses import replace
from decimal import Decimal

from app.claim_evidence.extraction import extract_supported_claims
from app.claim_evidence.models import ClaimEvidenceState
from app.claim_evidence.service import assess_claim_against_revenue
from app.retrieval.vector_store import VectorSearchResult
from app.temporal.financial import calculate_temporal_numeric_change
from app.temporal.models import (
    SourcePeriod,
    TemporalEvidence,
    TemporalFinancialValue,
)
from tests.test_claim_evidence_extraction import _source


def _filing_result(document_id: int, year: int, amount: str) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=100 + document_id,
        text=f"Total net sales {amount}",
        page_number=29,
        document_id=document_id,
        company="Apple",
        document_type="10-K",
        fiscal_year=year,
        cosine_distance=Decimal("0.1"),
        similarity=Decimal("0.9"),
    )


def _calculation(newer_amount: str = "416161"):
    older = TemporalFinancialValue(
        metric="total_net_sales",
        amount=Decimal("391035"),
        unit="USD millions",
        fiscal_year=2024,
        evidence=TemporalEvidence(
            1,
            SourcePeriod.OLDER,
            _filing_result(2, 2024, "$391,035 million"),
        ),
    )
    newer = TemporalFinancialValue(
        metric="total_net_sales",
        amount=Decimal(newer_amount),
        unit="USD millions",
        fiscal_year=2025,
        evidence=TemporalEvidence(
            2,
            SourcePeriod.NEWER,
            _filing_result(1, 2025, f"${newer_amount} million"),
        ),
    )
    return calculate_temporal_numeric_change(older, newer)


def _claims():
    return {claim.claim_id: claim for claim in extract_supported_claims(_source())}


def test_numeric_claim_supported_with_display_precision_tolerance() -> None:
    assessment = assess_claim_against_revenue(
        _claims()["fy2025_revenue_amount"],
        temporal_calculation=_calculation(),
    )
    assert assessment.state == ClaimEvidenceState.SUPPORTED
    assert assessment.actual_amount_millions == Decimal("416161")
    assert assessment.rounding_tolerance_millions == Decimal("500")


def test_numeric_claim_contradicted_outside_rounding_tolerance() -> None:
    claim = replace(
        _claims()["fy2025_revenue_amount"],
        stated_amount_millions=Decimal("500000"),
        normalized_text="Fiscal 2025 revenue reached $500 billion.",
    )
    assessment = assess_claim_against_revenue(
        claim, temporal_calculation=_calculation()
    )
    assert assessment.state == ClaimEvidenceState.CONTRADICTED


def test_compound_claim_is_only_partially_supported() -> None:
    assessment = assess_claim_against_revenue(
        _claims()["record_year_compound_performance"],
        temporal_calculation=_calculation(),
    )
    assert assessment.state == ClaimEvidenceState.PARTIALLY_SUPPORTED
    assert "all-time record" in assessment.rationale


def test_ambiguous_eps_basis_is_not_guessed() -> None:
    assessment = assess_claim_against_revenue(
        _claims()["double_digit_eps_growth"]
    )
    assert assessment.state == ClaimEvidenceState.AMBIGUOUS


def test_missing_installed_base_evidence_is_explicit() -> None:
    assessment = assess_claim_against_revenue(_claims()["installed_base_record"])
    assert assessment.state == ClaimEvidenceState.INSUFFICIENT_EVIDENCE


def test_subjective_claim_is_not_forced_into_binary_state() -> None:
    assessment = assess_claim_against_revenue(
        _claims()["customer_satisfaction_and_loyalty"]
    )
    assert assessment.state == ClaimEvidenceState.NOT_OBJECTIVELY_VERIFIABLE


def test_missing_filing_inputs_return_insufficient_evidence() -> None:
    assessment = assess_claim_against_revenue(_claims()["fy2025_revenue_amount"])
    assert assessment.state == ClaimEvidenceState.INSUFFICIENT_EVIDENCE
    assert assessment.filing_evidence == ()


def test_cross_period_arithmetic_and_provenance_are_preserved() -> None:
    calculation = _calculation()
    assessment = assess_claim_against_revenue(
        _claims()["fy2025_revenue_amount"],
        temporal_calculation=calculation,
    )
    assert assessment.temporal_calculation is not calculation
    assert assessment.temporal_calculation.percentage_change == Decimal("6.4")
    assert assessment.temporal_calculation.absolute_change == Decimal("25126")
    assert [item.evidence_id for item in assessment.filing_evidence] == [2, 3]
    assert [item.result.document_id for item in assessment.filing_evidence] == [2, 1]


def test_rounding_boundary_is_deterministic() -> None:
    claim = _claims()["fy2025_revenue_amount"]
    supported = assess_claim_against_revenue(
        claim, temporal_calculation=_calculation("416500")
    )
    contradicted = assess_claim_against_revenue(
        claim, temporal_calculation=_calculation("416501")
    )
    assert supported.state == ClaimEvidenceState.SUPPORTED
    assert contradicted.state == ClaimEvidenceState.CONTRADICTED
