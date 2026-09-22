"""Grounded explanation of an authoritative claim-vs-evidence assessment."""

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.claim_evidence.models import (
    ClaimEvidenceAnalysis,
    ClaimEvidenceAssessment,
    ClaimEvidenceState,
)
from app.generation.llm_client import LLMClient
from app.generation.prompts import SYSTEM_PROMPT, build_grounded_user_prompt
from app.generation.service import (
    CitationValidationError,
    GroundedAnswer,
    GroundedGenerationResult,
    StructuredResponseError,
    parse_and_validate_answer,
)


CLAIM_EVIDENCE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are explaining an application-produced management-claim assessment.
Keep four layers separate: the attributed claim, filing evidence, deterministic
analysis, and interpretation. The supplied classification is authoritative.
Return it unchanged, do not recompute financial values, and do not accuse anyone
of lying, deception, or bad intent. Say only what the available evidence supports.
When filing evidence exists, cite both the management source and filing evidence.
For qualitative or unavailable claims, cite the management source and explain the
limitation without inventing a filing fact.
Every paraphrase of management's statement must carry [1] inline. Before returning,
check that citation_ids exactly matches the unique inline markers in the summaries.
Keep the response fields separate: claim_summary states only management's claim
and ends with [1]; filing_summary states only an underlying filing fact and ends
with its filing marker, or is null when no filing evidence exists; interpretation
explains the authoritative application classification without citation markers.
Do not mix deterministic rounding/classification conclusions into either summary.
The structured insufficient_evidence field describes whether any grounded response
can be produced, not the deterministic classification label. When the supplied
classification is insufficient_evidence or not_objectively_verifiable but the
management quote is present, set the field to false, cite that quote, and explain
the evidence limitation.
"""


class ClaimEvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    claim_summary: str = Field(min_length=1)
    filing_summary: str | None
    interpretation: str = Field(min_length=1)
    classification: ClaimEvidenceState
    citation_ids: list[int]
    insufficient_evidence: bool


def _parse_response(
    raw: str,
    *,
    assessment: ClaimEvidenceAssessment,
) -> GroundedAnswer:
    try:
        response = ClaimEvidenceResponse.model_validate_json(raw)
    except ValidationError as exc:
        raise StructuredResponseError(
            "LLM response did not match the claim-evidence schema"
        ) from exc
    if response.classification != assessment.state:
        raise CitationValidationError(
            "LLM classification attempted to override deterministic assessment"
        )
    claim_ids = {int(value) for value in re.findall(r"\[(\d+)\]", response.claim_summary)}
    if claim_ids != {1}:
        raise CitationValidationError("claim_summary must cite only evidence ID 1")
    _validate_citation_sentence_boundaries(response.claim_summary)

    filing_ids = {item.evidence_id for item in assessment.filing_evidence}
    if filing_ids:
        if response.filing_summary is None:
            raise CitationValidationError(
                "filing_summary is required when filing evidence exists"
            )
        returned_filing_ids = {
            int(value) for value in re.findall(r"\[(\d+)\]", response.filing_summary)
        }
        if not returned_filing_ids or not returned_filing_ids.issubset(filing_ids):
            raise CitationValidationError(
                "filing_summary must cite supplied filing evidence"
            )
        _validate_citation_sentence_boundaries(response.filing_summary)
    elif response.filing_summary is not None:
        raise CitationValidationError(
            "filing_summary must be null when no filing evidence exists"
        )
    if re.search(r"\[\d+\]", response.interpretation):
        raise CitationValidationError(
            "application interpretation must not contain source citation markers"
        )

    rendered = " ".join(
        part
        for part in (
            response.claim_summary.strip(),
            response.filing_summary.strip() if response.filing_summary else None,
            response.interpretation.strip(),
        )
        if part
    )
    answer = GroundedAnswer(
        answer=rendered,
        citation_ids=response.citation_ids,
        insufficient_evidence=response.insufficient_evidence,
    )
    validated = parse_and_validate_answer(
        answer.model_dump_json(), evidence_count=len(assessment.all_evidence)
    )
    if not validated.insufficient_evidence:
        returned = set(validated.citation_ids)
        if 1 not in returned:
            raise CitationValidationError(
                "claim-vs-evidence interpretation must cite the management source"
            )
        filing_ids = {item.evidence_id for item in assessment.filing_evidence}
        if filing_ids and not returned.intersection(filing_ids):
            raise CitationValidationError(
                "claim-vs-evidence interpretation must cite filing evidence"
            )
    return validated


def _validate_citation_sentence_boundaries(answer: str) -> None:
    """Keep source-backed claims separate from application interpretation."""
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        markers = list(re.finditer(r"\[\d+\]", sentence))
        if not markers:
            continue
        tail = sentence[markers[-1].end() :].strip(" \t\r\n.,;:!?")
        if tail:
            raise CitationValidationError(
                "citation-bearing sentences must end at their final citation marker"
            )


class ClaimEvidenceGenerationService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def interpret(
        self, *, assessment: ClaimEvidenceAssessment
    ) -> ClaimEvidenceAnalysis:
        evidence = assessment.all_evidence
        user_prompt = build_grounded_user_prompt(
            "What did management claim, and does the available filing "
            "evidence support that claim?",
            [item.result for item in evidence],
            deterministic_calculation=assessment.as_prompt_block(),
        )
        raw = self._llm_client.generate(
            system_prompt=CLAIM_EVIDENCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=ClaimEvidenceResponse,
        )
        try:
            answer = _parse_response(raw, assessment=assessment)
        except (CitationValidationError, StructuredResponseError) as exc:
            repair_prompt = (
                f"{user_prompt}\n\n"
                "Your previous JSON response failed deterministic validation. "
                "Return one corrected JSON object only. Preserve the authoritative "
                "classification and factual content. Do not add facts. Ensure [1] "
                "ends claim_summary, include a supplied filing marker at the end of "
                "filing_summary when filing evidence exists, and make citation_ids "
                "exactly equal those summary markers. Put deterministic conclusions "
                "only in the uncited interpretation field.\n"
                f"Validation error: {exc}\n"
                f"Rejected response:\n{raw}"
            )
            corrected = self._llm_client.generate(
                system_prompt=CLAIM_EVIDENCE_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                response_format=ClaimEvidenceResponse,
            )
            answer = _parse_response(corrected, assessment=assessment)
        return ClaimEvidenceAnalysis(
            assessment=assessment,
            interpretation=GroundedGenerationResult(answer, evidence),
        )
