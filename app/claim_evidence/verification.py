"""M7 verification adapter for M10 claim-source and filing provenance."""

import re
from dataclasses import dataclass

from app.claim_evidence.models import ClaimEvidenceAnalysis
from app.generation.service import (
    GroundedAnswer,
    GroundedGenerationResult,
    NumberedEvidence,
)
from app.verification.service import (
    AnswerVerificationStatus,
    CitationVerificationReport,
    CitationVerificationService,
)


@dataclass(frozen=True)
class ClaimEvidenceVerification:
    analysis: ClaimEvidenceAnalysis
    claim_source: CitationVerificationReport
    filing_sources: CitationVerificationReport | None
    interpretation: CitationVerificationReport
    calculation_preserved: bool = True

    @property
    def fully_verified(self) -> bool:
        return (
            self.claim_source.status == AnswerVerificationStatus.VERIFIED
            and self.filing_sources is not None
            and self.filing_sources.status == AnswerVerificationStatus.VERIFIED
            and self.interpretation.status == AnswerVerificationStatus.VERIFIED
        )


class ClaimEvidenceVerificationService:
    def __init__(self, citation_verifier: CitationVerificationService) -> None:
        self._citation_verifier = citation_verifier

    def verify(self, analysis: ClaimEvidenceAnalysis) -> ClaimEvidenceVerification:
        assessment = analysis.assessment
        claim = assessment.claim
        source_hypothesis = claim.original_text
        if source_hypothesis.casefold().startswith("revenue reaching"):
            source_hypothesis = re.sub(
                r"^revenue reaching",
                "Management claimed that revenue reached",
                source_hypothesis,
                flags=re.IGNORECASE,
            )
        else:
            source_hypothesis = f"Management stated: {source_hypothesis}"
        claim_generation = GroundedGenerationResult(
            answer=GroundedAnswer(
                answer=f"{source_hypothesis} [1].",
                citation_ids=[1],
                insufficient_evidence=False,
            ),
            evidence=(assessment.claim_source_evidence,),
        )
        claim_report = self._citation_verifier.verify_generation(claim_generation)

        filing_report: CitationVerificationReport | None = None
        if assessment.temporal_calculation is not None:
            newer = assessment.temporal_calculation.newer
            filing_generation = GroundedGenerationResult(
                answer=GroundedAnswer(
                    answer=(
                        f"Apple's {newer.fiscal_year} 10-K reports total net sales "
                        f"of ${newer.amount:,} million [{newer.evidence.evidence_id}]."
                    ),
                    citation_ids=[newer.evidence.evidence_id],
                    insufficient_evidence=False,
                ),
                evidence=assessment.filing_evidence,
            )
            filing_report = self._citation_verifier.verify_generation(
                filing_generation
            )

        interpretation_report = self._citation_verifier.verify_generation(
            analysis.interpretation
        )
        return ClaimEvidenceVerification(
            analysis=analysis,
            claim_source=claim_report,
            filing_sources=filing_report,
            interpretation=interpretation_report,
        )
