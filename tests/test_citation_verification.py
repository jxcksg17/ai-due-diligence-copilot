"""Tests for claim-level citation entailment verification."""

from decimal import Decimal

import pytest

from app.financial.revenue_growth import RevenueGrowthResult, SourceFinancialValue
from app.financial.service import RevenueGrowthAnalysis
from app.generation.service import (
    CitationValidationError,
    GroundedAnswer,
    GroundedGenerationResult,
    NumberedEvidence,
)
from app.retrieval.vector_store import VectorSearchResult
from app.verification.service import (
    AnswerVerificationStatus,
    CitationVerificationService,
    VerificationStatus,
)
from app.verification.verifier import EntailmentProbabilities


SUPPORTED = EntailmentProbabilities(contradiction=0.05, entailment=0.9, neutral=0.05)
UNSUPPORTED = EntailmentProbabilities(contradiction=0.8, entailment=0.05, neutral=0.15)
AMBIGUOUS = EntailmentProbabilities(contradiction=0.1, entailment=0.3, neutral=0.6)


def _evidence(evidence_id: int, chunk_id: int, text: str) -> NumberedEvidence:
    return NumberedEvidence(
        evidence_id=evidence_id,
        result=VectorSearchResult(
            chunk_id=chunk_id,
            text=text,
            page_number=chunk_id,
            document_id=1,
            company="Apple",
            document_type="10-K",
            fiscal_year=2025,
            cosine_distance=0.2,
            similarity=0.8,
        ),
    )


def _generation(
    answer: str,
    citation_ids: list[int],
    evidence: tuple[NumberedEvidence, ...],
    *,
    insufficient: bool = False,
) -> GroundedGenerationResult:
    return GroundedGenerationResult(
        answer=GroundedAnswer(
            answer=answer,
            citation_ids=citation_ids,
            insufficient_evidence=insufficient,
        ),
        evidence=evidence,
    )


class FakeVerifier:
    model_name = "fake-nli"

    def __init__(
        self,
        predictions: list[EntailmentProbabilities] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.predictions = predictions or []
        self.error = error
        self.pairs: list[tuple[str, str]] = []

    def verify_many(
        self, pairs: list[tuple[str, str]]
    ) -> list[EntailmentProbabilities]:
        self.pairs = pairs
        if self.error is not None:
            raise self.error
        return self.predictions


def test_clearly_supported_claim_is_verified_with_provenance() -> None:
    verifier = FakeVerifier([SUPPORTED])
    generation = _generation(
        "Apple reported total net sales of $416,161 million [1].",
        [1],
        (_evidence(1, 141, "Total net sales $416,161 million."),),
    )
    report = CitationVerificationService(verifier).verify_generation(generation)
    claim = report.claims[0]
    assert report.status == AnswerVerificationStatus.VERIFIED
    assert claim.status == VerificationStatus.SUPPORTED
    assert claim.probabilities == SUPPORTED
    assert claim.evidence[0].citation_id == 1
    assert claim.evidence[0].chunk_id == 141
    assert claim.evidence[0].page_number == 141
    assert claim.evidence[0].document_id == 1


def test_clearly_unsupported_claim_is_flagged_without_rewriting_answer() -> None:
    generation = _generation(
        "Apple reported total net sales of $999 million [1].",
        [1],
        (_evidence(1, 141, "Total net sales were $416,161 million."),),
    )
    report = CitationVerificationService(FakeVerifier([UNSUPPORTED])).verify_generation(
        generation
    )
    assert report.status == AnswerVerificationStatus.FLAGGED
    assert report.claims[0].status == VerificationStatus.UNSUPPORTED
    assert report.generation.answer.answer == generation.answer.answer
    assert report.warnings


def test_wrong_existing_citation_to_unrelated_evidence_is_flagged() -> None:
    generation = _generation(
        "Research and development expense was $34,550 million [2].",
        [2],
        (
            _evidence(1, 141, "Research and development expense was $34,550 million."),
            _evidence(2, 57, "The company is exposed to component shortages."),
        ),
    )
    report = CitationVerificationService(FakeVerifier([UNSUPPORTED])).verify_generation(
        generation
    )
    assert report.claims[0].citation_ids == (2,)
    assert report.claims[0].status == VerificationStatus.UNSUPPORTED


def test_multi_citation_claim_is_verified_against_combined_evidence() -> None:
    verifier = FakeVerifier([SUPPORTED])
    generation = _generation(
        "Apple faces manufacturing concentration and component shortages [1] [2].",
        [1, 2],
        (
            _evidence(1, 55, "Manufacturing depends on outsourcing partners."),
            _evidence(2, 57, "Components are subject to industry-wide shortages."),
        ),
    )
    report = CitationVerificationService(verifier).verify_generation(generation)
    assert report.claims[0].status == VerificationStatus.SUPPORTED
    assert "[EVIDENCE 1]" in verifier.pairs[0][0]
    assert "[EVIDENCE 2]" in verifier.pairs[0][0]
    assert "outsourcing partners" in verifier.pairs[0][0]
    assert "industry-wide shortages" in verifier.pairs[0][0]
    assert verifier.pairs[0][1] == (
        "Apple faces manufacturing concentration and component shortages."
    )


def test_verification_excerpt_preserves_table_header_and_matching_row() -> None:
    verifier = FakeVerifier([SUPPORTED])
    generation = _generation(
        "Apple's total net sales in 2025 were $416,161 million [1].",
        [1],
        (
            _evidence(
                1,
                185,
                "The following table shows net sales for 2025 and 2024 (in millions):\n"
                "2025 2024\n"
                "iPhone $209,586 $201,183\n"
                "Total net sales $416,161 $391,035",
            ),
        ),
    )
    CitationVerificationService(verifier).verify_generation(generation)
    premise = verifier.pairs[0][0]
    assert "for 2025 and 2024 (in millions)" in premise
    assert "Total net sales $416,161 $391,035" in premise


def test_verification_excerpt_repairs_pdf_line_wraps_in_prose() -> None:
    verifier = FakeVerifier([SUPPORTED])
    generation = _generation(
        "Apple depends on outsourcing partners for manufacturing [1].",
        [1],
        (
            _evidence(
                1,
                54,
                "The Company depends on component and product manufacturing\n"
                "and logistical services provided by outsourcing partners.",
            ),
        ),
    )
    CitationVerificationService(verifier).verify_generation(generation)
    assert (
        "component and product manufacturing and logistical services"
        in verifier.pairs[0][0]
    )


def test_partially_supported_claim_is_ambiguous_and_flagged() -> None:
    generation = _generation(
        "Apple faces component shortages and permanently closed factories [1].",
        [1],
        (_evidence(1, 57, "Components are subject to industry-wide shortages."),),
    )
    report = CitationVerificationService(FakeVerifier([AMBIGUOUS])).verify_generation(
        generation
    )
    assert report.claims[0].status == VerificationStatus.AMBIGUOUS
    assert report.status == AnswerVerificationStatus.FLAGGED


def test_insufficient_evidence_response_skips_verifier() -> None:
    verifier = FakeVerifier(error=AssertionError("must not be called"))
    generation = _generation(
        "The supplied evidence is insufficient to answer the question.",
        [],
        (),
        insufficient=True,
    )
    report = CitationVerificationService(verifier).verify_generation(generation)
    assert report.status == AnswerVerificationStatus.INSUFFICIENT_EVIDENCE
    assert report.claims == ()
    assert verifier.pairs == []


def test_unknown_citation_id_preserves_existing_id_validation() -> None:
    generation = _generation(
        "Claim [2].",
        [2],
        (_evidence(1, 1, "Evidence"),),
    )
    with pytest.raises(CitationValidationError, match="unsupplied"):
        CitationVerificationService(FakeVerifier([SUPPORTED])).verify_generation(
            generation
        )


def test_verifier_failure_is_explicit_and_fails_closed() -> None:
    generation = _generation(
        "Supported-looking claim [1].",
        [1],
        (_evidence(1, 1, "Evidence"),),
    )
    report = CitationVerificationService(
        FakeVerifier(error=RuntimeError("model unavailable"))
    ).verify_generation(generation)
    assert report.status == AnswerVerificationStatus.FLAGGED
    assert report.claims[0].status == VerificationStatus.ERROR
    assert report.claims[0].error == "model unavailable"


def test_financial_verification_checks_sources_without_changing_calculation() -> None:
    evidence = (_evidence(1, 141, "Total net sales were 110 in 2025 and 100 in 2024."),)
    common = {
        "metric": "total_net_sales",
        "unit": "USD millions",
        "evidence_id": 1,
        "chunk_id": 141,
        "page_number": 141,
        "document_id": 1,
    }
    calculation = RevenueGrowthResult(
        current=SourceFinancialValue(fiscal_year=2025, amount=Decimal("110"), **common),
        prior=SourceFinancialValue(fiscal_year=2024, amount=Decimal("100"), **common),
        percentage_change=Decimal("10.0"),
        rounding_places=1,
    )
    generation = _generation(
        "Total net sales grew 10.0% [1].",
        [1],
        evidence,
    )
    analysis = RevenueGrowthAnalysis(
        calculation=calculation,
        interpretation=generation,
    )
    report = CitationVerificationService(
        FakeVerifier([SUPPORTED, SUPPORTED])
    ).verify_revenue_growth(analysis)
    assert report.status == AnswerVerificationStatus.VERIFIED
    assert report.calculation_preserved is True
    assert report.analysis.calculation is calculation
    assert [claim.evidence[0].chunk_id for claim in report.source_claims] == [141, 141]
