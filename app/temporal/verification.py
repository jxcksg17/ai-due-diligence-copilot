"""M7 verification adapters for M8 temporal analyses."""

from dataclasses import dataclass

from app.generation.service import GroundedAnswer, GroundedGenerationResult, NumberedEvidence
from app.temporal.generation import (
    TemporalDisclosureAnalysis,
    TemporalFinancialAnalysis,
)
from app.verification.service import (
    CitationVerificationReport,
    CitationVerificationService,
)


@dataclass(frozen=True)
class TemporalFinancialVerification:
    analysis: TemporalFinancialAnalysis
    source_verification: CitationVerificationReport
    calculation_preserved: bool = True


class TemporalVerificationService:
    """Verify temporal source claims while preserving deterministic results."""

    def __init__(self, citation_verifier: CitationVerificationService) -> None:
        self._citation_verifier = citation_verifier

    def verify_financial(
        self,
        analysis: TemporalFinancialAnalysis,
    ) -> TemporalFinancialVerification:
        comparison = analysis.comparison
        older = comparison.older
        newer = comparison.newer
        source_answer = GroundedAnswer(
            answer=(
                f"The {older.evidence.result.company} {older.fiscal_year} "
                f"{older.evidence.result.document_type} reports total net sales of "
                f"${older.amount:,} million [{older.evidence.evidence_id}]. "
                f"The {newer.evidence.result.company} {newer.fiscal_year} "
                f"{newer.evidence.result.document_type} reports total net sales of "
                f"${newer.amount:,} million [{newer.evidence.evidence_id}]."
            ),
            citation_ids=[older.evidence.evidence_id, newer.evidence.evidence_id],
            insufficient_evidence=False,
        )
        source_generation = GroundedGenerationResult(
            answer=source_answer,
            evidence=tuple(
                NumberedEvidence(
                    evidence_id=value.evidence.evidence_id,
                    result=value.evidence.result,
                )
                for value in (older, newer)
            ),
        )
        report = self._citation_verifier.verify_generation(source_generation)
        return TemporalFinancialVerification(
            analysis=analysis,
            source_verification=report,
        )

    def verify_disclosure(
        self,
        analysis: TemporalDisclosureAnalysis,
    ) -> CitationVerificationReport:
        return self._citation_verifier.verify_generation(analysis.interpretation)
