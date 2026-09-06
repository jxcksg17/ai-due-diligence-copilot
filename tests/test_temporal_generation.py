"""Grounded temporal generation and M7 integration tests."""

from decimal import Decimal

import pytest

from app.generation.service import CitationValidationError, GroundedAnswer
from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.temporal.generation import TemporalGenerationService
from app.temporal.models import (
    DisclosureComparison,
    DisclosurePeriodEvidence,
    DisclosureTopic,
    SourcePeriod,
    TemporalChangeCategory,
    TemporalEvidence,
    TemporalFinancialValue,
)
from app.temporal.financial import calculate_temporal_numeric_change
from app.temporal.resolution import TemporalDocumentPair
from app.temporal.verification import TemporalVerificationService
from app.verification.service import AnswerVerificationStatus, CitationVerificationService
from app.verification.verifier import EntailmentProbabilities


SUPPORTED = EntailmentProbabilities(contradiction=0.02, entailment=0.95, neutral=0.03)


def _result(evidence_id: int, document_id: int, year: int, text: str) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=100 + evidence_id,
        text=text,
        page_number=30 + evidence_id,
        document_id=document_id,
        company="Apple",
        document_type="10-K",
        fiscal_year=year,
        cosine_distance=0.1,
        similarity=0.9,
    )


def _documents() -> TemporalDocumentPair:
    return TemporalDocumentPair(
        older=DocumentMetadata(2, "Apple", "10-K", 2024),
        newer=DocumentMetadata(1, "Apple", "10-K", 2025),
    )


def _numeric_change():
    older_evidence = TemporalEvidence(
        1,
        SourcePeriod.OLDER,
        _result(1, 2, 2024, "Total net sales in 2024 were $100 million."),
    )
    newer_evidence = TemporalEvidence(
        2,
        SourcePeriod.NEWER,
        _result(2, 1, 2025, "Total net sales in 2025 were $110 million."),
    )
    return calculate_temporal_numeric_change(
        TemporalFinancialValue(
            "total_net_sales", Decimal("100"), "USD millions", 2024, older_evidence
        ),
        TemporalFinancialValue(
            "total_net_sales", Decimal("110"), "USD millions", 2025, newer_evidence
        ),
    )


class FakeLLM:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer = answer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer.model_dump_json()


def test_financial_interpretation_requires_and_preserves_both_periods() -> None:
    llm = FakeLLM(
        GroundedAnswer(
            answer="Net sales rose from $100 million [1] to $110 million [2].",
            citation_ids=[1, 2],
            insufficient_evidence=False,
        )
    )
    analysis = TemporalGenerationService(llm).interpret_financial(
        question="How did net sales change?",
        documents=_documents(),
        comparison=_numeric_change(),
    )
    assert analysis.comparison.percentage_change == Decimal("10.0")
    assert analysis.interpretation.answer.citation_ids == [1, 2]
    prompt = llm.calls[0]["user_prompt"]
    assert "Deterministic temporal calculation" in prompt
    assert "document_id=2" in prompt and "document_id=1" in prompt


def test_temporal_answer_missing_one_period_citation_is_rejected() -> None:
    llm = FakeLLM(
        GroundedAnswer(
            answer="Net sales were $110 million [2].",
            citation_ids=[2],
            insufficient_evidence=False,
        )
    )
    with pytest.raises(CitationValidationError, match="every available period"):
        TemporalGenerationService(llm).interpret_financial(
            question="How did net sales change?",
            documents=_documents(),
            comparison=_numeric_change(),
        )


def _ambiguous_disclosure() -> DisclosureComparison:
    topic = DisclosureTopic("test", "test disclosure", (("test",),))
    older = DisclosurePeriodEvidence(2024, SourcePeriod.OLDER, (), True)
    newer = DisclosurePeriodEvidence(2025, SourcePeriod.NEWER, (), True)
    return DisclosureComparison(
        topic=topic,
        documents=_documents(),
        older=older,
        newer=newer,
        category=TemporalChangeCategory.AMBIGUOUS,
    )


def test_ambiguous_disclosure_returns_insufficient_without_calling_llm() -> None:
    llm = FakeLLM(
        GroundedAnswer(answer="must not run", citation_ids=[], insufficient_evidence=True)
    )
    analysis = TemporalGenerationService(llm).interpret_disclosure(
        question="What changed?",
        comparison=_ambiguous_disclosure(),
    )
    assert analysis.interpretation.answer.insufficient_evidence is True
    assert llm.calls == []


class FakeNLI:
    model_name = "fake-nli"

    def __init__(self, predictions) -> None:
        self.predictions = predictions

    def verify_many(self, pairs):
        return self.predictions[: len(pairs)]


def test_m7_verifies_temporal_financial_sources_without_changing_calculation() -> None:
    change = _numeric_change()
    llm = FakeLLM(
        GroundedAnswer(
            answer="Net sales increased 10.0% [1] [2].",
            citation_ids=[1, 2],
            insufficient_evidence=False,
        )
    )
    analysis = TemporalGenerationService(llm).interpret_financial(
        question="How did net sales change?",
        documents=_documents(),
        comparison=change,
    )
    verifier = TemporalVerificationService(
        CitationVerificationService(FakeNLI([SUPPORTED, SUPPORTED]))
    )
    report = verifier.verify_financial(analysis)
    assert report.source_verification.status == AnswerVerificationStatus.VERIFIED
    assert report.calculation_preserved is True
    assert report.analysis.comparison is change
    assert report.source_verification.claims[0].claim_text.startswith(
        "The Apple 2024 10-K reports"
    )
    provenance = [
        claim.evidence[0].document_id for claim in report.source_verification.claims
    ]
    assert provenance == [2, 1]


def test_m7_insufficient_disclosure_integration_skips_nli() -> None:
    analysis = TemporalGenerationService(
        FakeLLM(GroundedAnswer(answer="unused", citation_ids=[], insufficient_evidence=True))
    ).interpret_disclosure(
        question="What changed?",
        comparison=_ambiguous_disclosure(),
    )
    report = TemporalVerificationService(
        CitationVerificationService(FakeNLI([]))
    ).verify_disclosure(analysis)
    assert report.status == AnswerVerificationStatus.INSUFFICIENT_EVIDENCE
