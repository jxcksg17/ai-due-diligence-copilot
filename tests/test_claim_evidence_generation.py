"""Grounded M10 generation, citation, and M7 integration tests."""

import pytest

from app.claim_evidence.generation import (
    ClaimEvidenceGenerationService,
    ClaimEvidenceResponse,
)
from app.claim_evidence.models import ClaimEvidenceState
from app.claim_evidence.service import assess_claim_against_revenue
from app.claim_evidence.verification import ClaimEvidenceVerificationService
from app.generation.service import CitationValidationError
from app.verification.service import (
    AnswerVerificationStatus,
    CitationVerificationService,
    VerificationStatus,
)
from app.verification.verifier import EntailmentProbabilities
from tests.test_claim_evidence_extraction import extract_supported_claims, _source
from tests.test_claim_evidence_service import _calculation


SUPPORTED = EntailmentProbabilities(contradiction=0.02, entailment=0.95, neutral=0.03)


def _claims():
    return {claim.claim_id: claim for claim in extract_supported_claims(_source())}


def _supported_assessment():
    return assess_claim_against_revenue(
        _claims()["fy2025_revenue_amount"],
        temporal_calculation=_calculation(),
    )


class FakeLLM:
    def __init__(
        self,
        response: ClaimEvidenceResponse,
        *additional_responses: ClaimEvidenceResponse,
    ) -> None:
        self.responses = [response, *additional_responses]
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index].model_dump_json()


def _response(
    *,
    classification: ClaimEvidenceState = ClaimEvidenceState.SUPPORTED,
    claim_summary: str = "Management said revenue reached $416 billion [1].",
    filing_summary: str | None = "The 10-K reports $416,161 million [3].",
    interpretation: str = "The deterministic application assessment is supported.",
    citation_ids: list[int] | None = None,
) -> ClaimEvidenceResponse:
    return ClaimEvidenceResponse(
        claim_summary=claim_summary,
        filing_summary=filing_summary,
        interpretation=interpretation,
        classification=classification,
        citation_ids=citation_ids or [1, 3],
        insufficient_evidence=False,
    )


def test_generation_preserves_layers_and_requires_both_source_types() -> None:
    llm = FakeLLM(_response())
    analysis = ClaimEvidenceGenerationService(llm).interpret(
        assessment=_supported_assessment()
    )
    assert analysis.assessment.state == ClaimEvidenceState.SUPPORTED
    assert analysis.interpretation.answer.citation_ids == [1, 3]
    prompt = llm.calls[0]["user_prompt"]
    assert "Original attributed claim" in prompt
    assert "Classification: supported" in prompt
    assert "document_type=earnings_release" in prompt
    assert "document_type=10-K" in prompt
    assert "Keep the response fields separate" in llm.calls[0]["system_prompt"]


def test_missing_management_source_citation_is_rejected() -> None:
    llm = FakeLLM(
        _response(
            claim_summary="Management stated the revenue claim without a marker.",
            citation_ids=[3],
        )
    )
    with pytest.raises(CitationValidationError, match="evidence ID 1"):
        ClaimEvidenceGenerationService(llm).interpret(
            assessment=_supported_assessment()
        )


def test_missing_filing_citation_is_rejected() -> None:
    llm = FakeLLM(
        _response(
            filing_summary=None,
            citation_ids=[1],
        )
    )
    with pytest.raises(CitationValidationError, match="filing evidence"):
        ClaimEvidenceGenerationService(llm).interpret(
            assessment=_supported_assessment()
        )


def test_hallucinated_citation_id_is_rejected() -> None:
    llm = FakeLLM(
        _response(
            filing_summary="An unrelated source says something else [99].",
            citation_ids=[1, 99],
        )
    )
    with pytest.raises(CitationValidationError, match="supplied filing evidence"):
        ClaimEvidenceGenerationService(llm).interpret(
            assessment=_supported_assessment()
        )


def test_one_bounded_correction_pass_can_fix_citation_contract() -> None:
    invalid = _response(
        claim_summary="Management said revenue reached $416 billion.",
        citation_ids=[1, 3],
    )
    llm = FakeLLM(invalid, _response())
    analysis = ClaimEvidenceGenerationService(llm).interpret(
        assessment=_supported_assessment()
    )
    assert analysis.interpretation.answer.citation_ids == [1, 3]
    assert len(llm.calls) == 2
    assert "failed deterministic validation" in llm.calls[1]["user_prompt"]


def test_citation_sentence_cannot_mix_source_fact_and_application_conclusion() -> None:
    invalid = _response(
        filing_summary=(
            "The filing reports $416,161 million [3], which proves the classification."
        ),
    )
    llm = FakeLLM(invalid, _response())
    analysis = ClaimEvidenceGenerationService(llm).interpret(
        assessment=_supported_assessment()
    )
    assert len(llm.calls) == 2
    assert "The 10-K reports $416,161 million [3]." in (
        analysis.interpretation.answer.answer
    )


def test_qwen_cannot_override_deterministic_classification() -> None:
    llm = FakeLLM(_response(classification=ClaimEvidenceState.CONTRADICTED))
    with pytest.raises(CitationValidationError, match="override"):
        ClaimEvidenceGenerationService(llm).interpret(
            assessment=_supported_assessment()
        )


def test_not_objectively_verifiable_allows_source_only_explanation() -> None:
    assessment = assess_claim_against_revenue(
        _claims()["customer_satisfaction_and_loyalty"]
    )
    llm = FakeLLM(
        _response(
            classification=ClaimEvidenceState.NOT_OBJECTIVELY_VERIFIABLE,
            claim_summary=(
                "Management described customer satisfaction and loyalty as very high [1]."
            ),
            filing_summary=None,
            interpretation="The application has no objective filing benchmark.",
            citation_ids=[1],
        )
    )
    analysis = ClaimEvidenceGenerationService(llm).interpret(assessment=assessment)
    assert analysis.assessment.has_filing_evidence is False


class FakeVerifier:
    model_name = "fake-nli"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def verify_many(self, pairs):
        self.calls.append(pairs)
        if self.fail:
            raise RuntimeError("verifier unavailable")
        return [SUPPORTED for _ in pairs]


def test_m7_verifies_claim_source_filing_source_and_interpretation() -> None:
    analysis = ClaimEvidenceGenerationService(FakeLLM(_response())).interpret(
        assessment=_supported_assessment()
    )
    verifier = FakeVerifier()
    report = ClaimEvidenceVerificationService(
        CitationVerificationService(verifier)
    ).verify(analysis)
    assert report.claim_source.status == AnswerVerificationStatus.VERIFIED
    assert report.filing_sources.status == AnswerVerificationStatus.VERIFIED
    assert report.interpretation.status == AnswerVerificationStatus.VERIFIED
    assert report.calculation_preserved is True
    assert report.fully_verified is True
    assert verifier.calls[0][0][1] == (
        "Management claimed that revenue reached $416 billion."
    )


def test_missing_filing_provenance_is_not_marked_fully_verified() -> None:
    assessment = assess_claim_against_revenue(
        _claims()["customer_satisfaction_and_loyalty"]
    )
    analysis = ClaimEvidenceGenerationService(
        FakeLLM(
            _response(
                classification=ClaimEvidenceState.NOT_OBJECTIVELY_VERIFIABLE,
                claim_summary="Management described satisfaction as very high [1].",
                filing_summary=None,
                interpretation="No objective filing benchmark is available.",
                citation_ids=[1],
            )
        )
    ).interpret(assessment=assessment)
    report = ClaimEvidenceVerificationService(
        CitationVerificationService(FakeVerifier())
    ).verify(analysis)
    assert report.filing_sources is None
    assert report.fully_verified is False


def test_verifier_failure_is_explicit_and_does_not_change_state() -> None:
    analysis = ClaimEvidenceGenerationService(FakeLLM(_response())).interpret(
        assessment=_supported_assessment()
    )
    report = ClaimEvidenceVerificationService(
        CitationVerificationService(FakeVerifier(fail=True))
    ).verify(analysis)
    assert report.claim_source.claims[0].status == VerificationStatus.ERROR
    assert report.fully_verified is False
    assert report.analysis.assessment.state == ClaimEvidenceState.SUPPORTED
