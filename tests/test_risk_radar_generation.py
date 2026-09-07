"""Grounded Risk Radar interpretation and M7 integration tests."""

import pytest

from app.generation.service import CitationValidationError, GroundedAnswer
from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.risk_radar.generation import RiskRadarGenerationService
from app.risk_radar.models import (
    RiskComparison,
    RiskEvidence,
    RiskPresenceState,
    RiskSignal,
    RiskTemporalState,
)
from app.risk_radar.taxonomy import SUPPLY_CHAIN_MANUFACTURING
from app.risk_radar.verification import RiskRadarVerificationService
from app.temporal.models import SourcePeriod
from app.temporal.resolution import TemporalDocumentPair
from app.verification.service import (
    AnswerVerificationStatus,
    CitationVerificationService,
    VerificationStatus,
)
from app.verification.verifier import EntailmentProbabilities


SUPPORTED = EntailmentProbabilities(0.01, 0.96, 0.03)
NEUTRAL = EntailmentProbabilities(0.01, 0.09, 0.90)


def _document(document_id: int, year: int) -> DocumentMetadata:
    return DocumentMetadata(document_id, "Apple", "10-K", year)


def _signal(
    document_id: int,
    year: int,
    period: SourcePeriod,
    evidence_id: int,
) -> RiskSignal:
    document = _document(document_id, year)
    result = VectorSearchResult(
        chunk_id=100 + document_id,
        text=(
            "The Company relies on single or limited suppliers for components, "
            "which exposes it to supply shortages."
        ),
        page_number=11,
        document_id=document_id,
        company="Apple",
        document_type="10-K",
        fiscal_year=year,
        cosine_distance=None,
        similarity=None,
    )
    evidence = RiskEvidence(
        evidence_id,
        SUPPLY_CHAIN_MANUFACTURING.key,
        period,
        result,
    )
    return RiskSignal(
        SUPPLY_CHAIN_MANUFACTURING,
        document,
        period,
        RiskPresenceState.DISCLOSED,
        (evidence,),
        1,
    )


def _comparison(
    state: RiskTemporalState = RiskTemporalState.RECURRING_AMBIGUOUS,
) -> RiskComparison:
    older = _signal(2, 2024, SourcePeriod.OLDER, 1)
    newer = _signal(1, 2025, SourcePeriod.NEWER, 2)
    return RiskComparison(
        SUPPLY_CHAIN_MANUFACTURING,
        TemporalDocumentPair(older.document, newer.document),
        older,
        newer,
        state,
    )


class FakeLLM:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer = answer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer.model_dump_json()


def test_interpretation_is_separate_grounded_and_cites_both_periods() -> None:
    llm = FakeLLM(
        GroundedAnswer(
            answer=(
                "Interpretation: Apple disclosed reliance on limited suppliers "
                "in both periods [1][2]."
            ),
            citation_ids=[1, 2],
            insufficient_evidence=False,
        )
    )
    analysis = RiskRadarGenerationService(llm).interpret(
        question="Summarize the supply-chain risk across both filings.",
        comparison=_comparison(),
    )
    assert analysis.interpretation.answer.citation_ids == [1, 2]
    assert analysis.comparison.temporal_state == RiskTemporalState.RECURRING_AMBIGUOUS
    prompt = llm.calls[0]["user_prompt"]
    assert "Deterministic Risk Radar signal" in prompt
    assert "recurring_ambiguous" in prompt
    assert "Older-period evidence IDs: [1]" in prompt
    assert "Newer-period evidence IDs: [2]" in prompt
    assert "Do not repeat the internal temporal-state label" in prompt
    assert "document_id=2" in prompt and "document_id=1" in prompt


def test_invalid_hallucinated_or_missing_period_citation_is_rejected() -> None:
    hallucinated = FakeLLM(
        GroundedAnswer(
            answer="Interpretation: A risk was disclosed [3].",
            citation_ids=[3],
            insufficient_evidence=False,
        )
    )
    with pytest.raises(CitationValidationError, match="unsupplied evidence IDs"):
        RiskRadarGenerationService(hallucinated).interpret(
            question="Summarize the risk.",
            comparison=_comparison(),
        )

    one_period = FakeLLM(
        GroundedAnswer(
            answer="Interpretation: A risk was disclosed [1].",
            citation_ids=[1],
            insufficient_evidence=False,
        )
    )
    with pytest.raises(CitationValidationError, match="every disclosed period"):
        RiskRadarGenerationService(one_period).interpret(
            question="Summarize the risk.",
            comparison=_comparison(),
        )


def test_interpretation_is_structurally_separate_from_authoritative_signal() -> None:
    llm = FakeLLM(
        GroundedAnswer(
            answer="Apple disclosed supplier concentration [1][2].",
            citation_ids=[1, 2],
            insufficient_evidence=False,
        )
    )
    analysis = RiskRadarGenerationService(llm).interpret(
        question="Summarize the risk.",
        comparison=_comparison(),
    )
    assert analysis.interpretation.answer.answer.startswith("Apple disclosed")
    assert analysis.comparison.temporal_state == RiskTemporalState.RECURRING_AMBIGUOUS


def test_uncited_generated_tail_is_rejected_before_m7() -> None:
    llm = FakeLLM(
        GroundedAnswer(
            answer=(
                "Interpretation: Supplier concentration was disclosed [1][2]. "
                "This could disrupt production."
            ),
            citation_ids=[1, 2],
            insufficient_evidence=False,
        )
    )
    with pytest.raises(CitationValidationError, match="uncited text"):
        RiskRadarGenerationService(llm).interpret(
            question="Summarize the risk.",
            comparison=_comparison(),
        )


def test_insufficient_signal_skips_generation() -> None:
    llm = FakeLLM(
        GroundedAnswer(answer="unused", citation_ids=[], insufficient_evidence=True)
    )
    older_document = _document(2, 2024)
    newer_document = _document(1, 2025)
    older = RiskSignal(
        SUPPLY_CHAIN_MANUFACTURING,
        older_document,
        SourcePeriod.OLDER,
        RiskPresenceState.NOT_FOUND,
        (),
        0,
    )
    newer = RiskSignal(
        SUPPLY_CHAIN_MANUFACTURING,
        newer_document,
        SourcePeriod.NEWER,
        RiskPresenceState.NOT_FOUND,
        (),
        0,
    )
    comparison = RiskComparison(
        SUPPLY_CHAIN_MANUFACTURING,
        TemporalDocumentPair(older_document, newer_document),
        older,
        newer,
        RiskTemporalState.NOT_FOUND_BOTH,
    )
    analysis = RiskRadarGenerationService(llm).interpret(
        question="Summarize the risk.",
        comparison=comparison,
    )
    assert analysis.interpretation.answer.insufficient_evidence is True
    assert llm.calls == []


class FakeNLI:
    model_name = "fake-nli"

    def __init__(self, predictions=(), error=None) -> None:
        self.predictions = list(predictions)
        self.error = error

    def verify_many(self, pairs):
        if self.error:
            raise self.error
        return self.predictions[: len(pairs)]


def _generated_analysis():
    llm = FakeLLM(
        GroundedAnswer(
            answer=(
                "Interpretation: Reliance on limited suppliers can expose the "
                "Company to shortages [1][2]."
            ),
            citation_ids=[1, 2],
            insufficient_evidence=False,
        )
    )
    return RiskRadarGenerationService(llm).interpret(
        question="Why may this risk matter?",
        comparison=_comparison(),
    )


def test_m7_verification_preserves_risk_signal_and_provenance() -> None:
    analysis = _generated_analysis()
    report = RiskRadarVerificationService(
        CitationVerificationService(FakeNLI([SUPPORTED]))
    ).verify(analysis)
    assert report.status == AnswerVerificationStatus.VERIFIED
    assert report.claims[0].status == VerificationStatus.SUPPORTED
    assert [item.document_id for item in report.claims[0].evidence] == [2, 1]
    assert analysis.comparison.temporal_state == RiskTemporalState.RECURRING_AMBIGUOUS


@pytest.mark.parametrize(
    ("verifier", "expected_status"),
    [
        (FakeNLI([NEUTRAL]), VerificationStatus.AMBIGUOUS),
        (FakeNLI(error=RuntimeError("model unavailable")), VerificationStatus.ERROR),
    ],
)
def test_m7_ambiguous_and_failure_behavior(verifier, expected_status) -> None:
    report = RiskRadarVerificationService(
        CitationVerificationService(verifier)
    ).verify(_generated_analysis())
    assert report.status == AnswerVerificationStatus.FLAGGED
    assert report.claims[0].status == expected_status
    assert report.warnings
