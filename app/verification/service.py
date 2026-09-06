"""Claim-level citation verification and explicit failure reporting."""

import re
from dataclasses import dataclass, replace
from enum import StrEnum

from app.financial.service import RevenueGrowthAnalysis
from app.generation.service import (
    CitationValidationError,
    GroundedGenerationResult,
    NumberedEvidence,
)
from app.verification.claims import ExtractedClaim, extract_cited_claims
from app.verification.verifier import EntailmentProbabilities, EntailmentVerifier


class VerificationStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"


class AnswerVerificationStatus(StrEnum):
    VERIFIED = "verified"
    FLAGGED = "flagged"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class CitationEvidence:
    citation_id: int
    chunk_id: int
    page_number: int
    document_id: int
    company: str
    document_type: str
    fiscal_year: int
    text: str
    verification_excerpt: str


@dataclass(frozen=True)
class ClaimVerification:
    claim_text: str
    citation_ids: tuple[int, ...]
    evidence: tuple[CitationEvidence, ...]
    status: VerificationStatus
    probabilities: EntailmentProbabilities | None
    verifier_model: str
    error: str | None = None


@dataclass(frozen=True)
class CitationVerificationReport:
    generation: GroundedGenerationResult
    status: AnswerVerificationStatus
    claims: tuple[ClaimVerification, ...]
    warnings: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return self.status == AnswerVerificationStatus.VERIFIED


@dataclass(frozen=True)
class FinancialCitationVerificationReport:
    analysis: RevenueGrowthAnalysis
    status: AnswerVerificationStatus
    source_claims: tuple[ClaimVerification, ...]
    warnings: tuple[str, ...]
    calculation_preserved: bool = True


class CitationVerificationService:
    """Verify cited claims without modifying or regenerating the answer."""

    def __init__(self, verifier: EntailmentVerifier) -> None:
        self._verifier = verifier

    def verify_generation(
        self, generation: GroundedGenerationResult
    ) -> CitationVerificationReport:
        if generation.answer.insufficient_evidence:
            return CitationVerificationReport(
                generation=generation,
                status=AnswerVerificationStatus.INSUFFICIENT_EVIDENCE,
                claims=(),
                warnings=(),
            )

        claims = extract_cited_claims(generation.answer.answer)
        if not claims:
            return CitationVerificationReport(
                generation=generation,
                status=AnswerVerificationStatus.FLAGGED,
                claims=(),
                warnings=("No citation-bearing claims could be extracted.",),
            )
        verifications = self._verify_claims(claims, generation.evidence)
        return CitationVerificationReport(
            generation=generation,
            status=_overall_status(verifications),
            claims=verifications,
            warnings=_warnings(verifications),
        )

    def verify_revenue_growth(
        self, analysis: RevenueGrowthAnalysis
    ) -> FinancialCitationVerificationReport:
        calculation = analysis.calculation
        claims = [
            ExtractedClaim(
                claim_text=(
                    f"Total net sales in {source.fiscal_year} were "
                    f"${source.amount:,} million."
                ),
                citation_ids=(source.evidence_id,),
            )
            for source in (calculation.current, calculation.prior)
        ]
        verifications = self._verify_claims(
            claims,
            analysis.interpretation.evidence,
        )
        _validate_financial_provenance(analysis, verifications)
        return FinancialCitationVerificationReport(
            analysis=analysis,
            status=_overall_status(verifications),
            source_claims=verifications,
            warnings=_warnings(verifications),
        )

    def _verify_claims(
        self,
        claims: list[ExtractedClaim],
        numbered_evidence: tuple[NumberedEvidence, ...],
    ) -> tuple[ClaimVerification, ...]:
        evidence_by_id = {
            item.evidence_id: CitationEvidence(
                citation_id=item.evidence_id,
                chunk_id=item.result.chunk_id,
                page_number=item.result.page_number,
                document_id=item.result.document_id,
                company=item.result.company,
                document_type=item.result.document_type,
                fiscal_year=item.result.fiscal_year,
                text=item.result.text,
                verification_excerpt="",
            )
            for item in numbered_evidence
        }
        resolved: list[tuple[ExtractedClaim, tuple[CitationEvidence, ...]]] = []
        for claim in claims:
            invalid_ids = set(claim.citation_ids) - set(evidence_by_id)
            if invalid_ids:
                rendered = ", ".join(str(value) for value in sorted(invalid_ids))
                raise CitationValidationError(
                    f"claim referenced unsupplied evidence IDs: {rendered}"
                )
            excerpt_budget = max(300, 1800 // len(claim.citation_ids))
            evidence = tuple(
                replace(
                    evidence_by_id[value],
                    verification_excerpt=_select_verification_excerpt(
                        evidence_by_id[value].text,
                        claim.claim_text,
                        max_chars=excerpt_budget,
                    ),
                )
                for value in claim.citation_ids
            )
            resolved.append((claim, evidence))

        pairs = [
            (_combined_premise(evidence), claim.claim_text)
            for claim, evidence in resolved
        ]
        try:
            predictions = self._verifier.verify_many(pairs)
            if len(predictions) != len(resolved):
                raise RuntimeError("verifier returned an unexpected number of predictions")
        except Exception as exc:
            return tuple(
                ClaimVerification(
                    claim_text=claim.claim_text,
                    citation_ids=claim.citation_ids,
                    evidence=evidence,
                    status=VerificationStatus.ERROR,
                    probabilities=None,
                    verifier_model=self._verifier.model_name,
                    error=str(exc),
                )
                for claim, evidence in resolved
            )

        return tuple(
            ClaimVerification(
                claim_text=claim.claim_text,
                citation_ids=claim.citation_ids,
                evidence=evidence,
                status=_classify(prediction),
                probabilities=prediction,
                verifier_model=self._verifier.model_name,
            )
            for (claim, evidence), prediction in zip(
                resolved, predictions, strict=True
            )
        )


def _combined_premise(evidence: tuple[CitationEvidence, ...]) -> str:
    return "\n\n".join(
        f"[EVIDENCE {item.citation_id}]\n"
        f"Company: {item.company}. Document: {item.document_type}, "
        f"fiscal year {item.fiscal_year}.\n"
        f"{item.verification_excerpt}"
        for item in evidence
    )


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _select_verification_excerpt(
    text: str,
    claim: str,
    *,
    max_chars: int,
) -> str:
    """Select inspectable source segments relevant to a claim within NLI limits."""
    claim_terms = {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(claim)
        if token.casefold() not in _STOP_WORDS
    }
    # PDF extraction inserts line breaks inside prose, while tables use those same
    # line breaks as row boundaries. Consider both representations: compact lines
    # preserve table headers/rows, and normalized sentences repair wrapped prose.
    normalized_text = re.sub(r"\s+", " ", text).strip()
    line_segments = [line.strip() for line in text.splitlines() if line.strip()]
    sentence_segments = [
        segment.strip()
        for segment in _SENTENCE_BOUNDARY_PATTERN.split(normalized_text)
        if segment.strip() and len(segment) <= 600
    ]
    segments = list(dict.fromkeys([*line_segments, *sentence_segments]))
    ranked = []
    for index, segment in enumerate(segments):
        segment_terms = {
            token.casefold() for token in _TOKEN_PATTERN.findall(segment)
        }
        overlap = len(claim_terms & segment_terms)
        ranked.append((overlap, index, segment))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected: list[tuple[int, str]] = []
    used_chars = 0
    for overlap, index, segment in ranked:
        if overlap == 0 and selected:
            break
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        excerpt = segment[:remaining].strip()
        if excerpt:
            if any(
                excerpt in existing or existing in excerpt
                for _, existing in selected
            ):
                continue
            selected.append((index, excerpt))
            used_chars += len(excerpt) + 1
        if len(selected) >= 3:
            break
    if not selected:
        return text[:max_chars].strip()
    selected.sort(key=lambda item: item[0])
    return " ".join(segment for _, segment in selected)


def _classify(probabilities: EntailmentProbabilities) -> VerificationStatus:
    if probabilities.entailment > 0.5:
        return VerificationStatus.SUPPORTED
    if probabilities.contradiction > 0.5:
        return VerificationStatus.UNSUPPORTED
    return VerificationStatus.AMBIGUOUS


def _overall_status(
    verifications: tuple[ClaimVerification, ...]
) -> AnswerVerificationStatus:
    if verifications and all(
        item.status == VerificationStatus.SUPPORTED for item in verifications
    ):
        return AnswerVerificationStatus.VERIFIED
    return AnswerVerificationStatus.FLAGGED


def _warnings(verifications: tuple[ClaimVerification, ...]) -> tuple[str, ...]:
    return tuple(
        f"Claim {index} is {item.status.value}: {item.claim_text}"
        for index, item in enumerate(verifications, start=1)
        if item.status != VerificationStatus.SUPPORTED
    )


def _validate_financial_provenance(
    analysis: RevenueGrowthAnalysis,
    verifications: tuple[ClaimVerification, ...],
) -> None:
    sources = (analysis.calculation.current, analysis.calculation.prior)
    for source, verification in zip(sources, verifications, strict=True):
        evidence = verification.evidence[0]
        if (
            source.chunk_id != evidence.chunk_id
            or source.page_number != evidence.page_number
            or source.document_id != evidence.document_id
        ):
            raise CitationValidationError(
                "deterministic financial source provenance does not match cited evidence"
            )
