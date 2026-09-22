"""Deterministic M10 claim-to-filing comparison services."""

from dataclasses import replace
from decimal import Decimal

from sqlalchemy.orm import Session

from app.claim_evidence.models import (
    ClaimEvidenceAssessment,
    ClaimEvidenceState,
    ManagementClaim,
    ManagementClaimType,
)
from app.generation.service import NumberedEvidence
from app.retrieval.base import Retriever
from app.temporal.financial import (
    TemporalFinancialComparisonService,
    TemporalFinancialInputError,
)
from app.temporal.models import TemporalFinancialValue, TemporalNumericChange
from app.temporal.resolution import TemporalResolutionError


def _renumber_value(
    value: TemporalFinancialValue, evidence_id: int
) -> TemporalFinancialValue:
    return replace(
        value,
        evidence=replace(value.evidence, evidence_id=evidence_id),
    )


def assess_claim_against_revenue(
    claim: ManagementClaim,
    *,
    temporal_calculation: TemporalNumericChange | None = None,
) -> ClaimEvidenceAssessment:
    """Classify one claim without delegating arithmetic or state to the LLM."""
    if claim.claim_type == ManagementClaimType.QUALITATIVE:
        return ClaimEvidenceAssessment(
            claim=claim,
            state=ClaimEvidenceState.NOT_OBJECTIVELY_VERIFIABLE,
            rationale=(
                "The claim uses a qualitative judgment without an objective filing "
                "measure or threshold."
            ),
            filing_evidence=(),
        )
    if claim.claim_type == ManagementClaimType.OBJECTIVE_UNAVAILABLE:
        return ClaimEvidenceAssessment(
            claim=claim,
            state=ClaimEvidenceState.INSUFFICIENT_EVIDENCE,
            rationale=(
                "The statement is objectively testable, but the available 10-K "
                "evidence does not report the required installed-base measure."
            ),
            filing_evidence=(),
        )
    if claim.claim_type == ManagementClaimType.AMBIGUOUS_FINANCIAL:
        return ClaimEvidenceAssessment(
            claim=claim,
            state=ClaimEvidenceState.AMBIGUOUS,
            rationale=(
                "The claim does not state the comparison basis clearly enough to "
                "select GAAP versus adjusted EPS without guessing."
            ),
            filing_evidence=(),
        )
    if temporal_calculation is None:
        return ClaimEvidenceAssessment(
            claim=claim,
            state=ClaimEvidenceState.INSUFFICIENT_EVIDENCE,
            rationale="The required cross-period filing evidence was unavailable.",
            filing_evidence=(),
        )
    if claim.stated_amount_millions is None or claim.display_precision_millions is None:
        raise ValueError("revenue claim is missing its stated amount or precision")

    older = _renumber_value(temporal_calculation.older, 2)
    newer = _renumber_value(temporal_calculation.newer, 3)
    calculation = replace(temporal_calculation, older=older, newer=newer)
    filing_evidence = (
        NumberedEvidence(2, older.evidence.result),
        NumberedEvidence(3, newer.evidence.result),
    )
    tolerance = claim.display_precision_millions / Decimal("2")
    difference = abs(newer.amount - claim.stated_amount_millions)
    amount_matches = difference <= tolerance

    if not amount_matches:
        state = ClaimEvidenceState.CONTRADICTED
        rationale = (
            "The filing amount falls outside the tolerance implied by the claim's "
            "displayed precision."
        )
    elif claim.claim_type == ManagementClaimType.COMPOUND_PERFORMANCE:
        state = ClaimEvidenceState.PARTIALLY_SUPPORTED
        rationale = (
            "The revenue amount is supported, but two available annual filings do "
            "not establish an all-time record and this increment does not resolve "
            "the claim's separate EPS basis."
        )
    else:
        state = ClaimEvidenceState.SUPPORTED
        rationale = (
            "The filing amount is within the rounding tolerance implied by the "
            "management statement."
        )
    return ClaimEvidenceAssessment(
        claim=claim,
        state=state,
        rationale=rationale,
        filing_evidence=filing_evidence,
        temporal_calculation=calculation,
        actual_amount_millions=newer.amount,
        rounding_tolerance_millions=tolerance,
    )


class ClaimEvidenceService:
    """Reuse M8 retrieval/calculation, then apply conservative M10 states."""

    def __init__(self, *, retriever: Retriever) -> None:
        self._temporal = TemporalFinancialComparisonService(retriever=retriever)

    def assess(
        self,
        db: Session,
        *,
        claim: ManagementClaim,
        older_year: int = 2024,
        newer_year: int = 2025,
        top_k: int = 10,
    ) -> ClaimEvidenceAssessment:
        if claim.claim_type not in {
            ManagementClaimType.REVENUE_AMOUNT,
            ManagementClaimType.COMPOUND_PERFORMANCE,
        }:
            return assess_claim_against_revenue(claim)
        try:
            _, calculation = self._temporal.compare_total_net_sales(
                db,
                company=claim.source.company,
                older_year=older_year,
                newer_year=newer_year,
                document_type="10-K",
                top_k=top_k,
            )
        except (TemporalFinancialInputError, TemporalResolutionError):
            return assess_claim_against_revenue(claim)
        return assess_claim_against_revenue(
            claim,
            temporal_calculation=calculation,
        )
