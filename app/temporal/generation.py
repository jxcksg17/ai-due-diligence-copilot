"""Grounded interpretation of already-computed temporal comparisons."""

from dataclasses import dataclass

from app.generation.llm_client import LLMClient
from app.generation.prompts import SYSTEM_PROMPT, build_grounded_user_prompt
from app.generation.service import (
    CitationValidationError,
    GroundedAnswer,
    GroundedGenerationResult,
    NumberedEvidence,
    parse_and_validate_answer,
)
from app.temporal.models import (
    DisclosureComparison,
    TemporalChangeCategory,
    TemporalNumericChange,
)
from app.temporal.resolution import TemporalDocumentPair


TEMPORAL_SYSTEM_PROMPT = SYSTEM_PROMPT + """

For temporal questions, distinguish the older and newer filing explicitly.
Use the supplied application comparison result as an alignment/calculation result;
do not recompute it or turn wording differences into claims of business materiality.
When evidence exists in both periods, cite at least one evidence block from each period.
"""


@dataclass(frozen=True)
class TemporalFinancialAnalysis:
    question: str
    documents: TemporalDocumentPair
    comparison: TemporalNumericChange
    interpretation: GroundedGenerationResult


@dataclass(frozen=True)
class TemporalDisclosureAnalysis:
    question: str
    comparison: DisclosureComparison
    interpretation: GroundedGenerationResult


class TemporalGenerationService:
    """Generate only after temporal resolution/alignment has completed."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def interpret_financial(
        self,
        *,
        question: str,
        documents: TemporalDocumentPair,
        comparison: TemporalNumericChange,
    ) -> TemporalFinancialAnalysis:
        evidence = tuple(
            NumberedEvidence(
                evidence_id=value.evidence.evidence_id,
                result=value.evidence.result,
            )
            for value in (comparison.older, comparison.newer)
        )
        interpretation = self._generate(
            question=question,
            evidence=evidence,
            comparison_block=comparison.as_prompt_block(),
            required_period_ids=(
                {comparison.older.evidence.evidence_id},
                {comparison.newer.evidence.evidence_id},
            ),
        )
        return TemporalFinancialAnalysis(
            question=question,
            documents=documents,
            comparison=comparison,
            interpretation=interpretation,
        )

    def interpret_disclosure(
        self,
        *,
        question: str,
        comparison: DisclosureComparison,
    ) -> TemporalDisclosureAnalysis:
        all_evidence = (*comparison.older.evidence, *comparison.newer.evidence)
        evidence = tuple(
            NumberedEvidence(evidence_id=item.evidence_id, result=item.result)
            for item in all_evidence
        )
        if comparison.category in {
            TemporalChangeCategory.AMBIGUOUS,
            TemporalChangeCategory.UNAVAILABLE,
        }:
            interpretation = GroundedGenerationResult(
                answer=GroundedAnswer(
                    answer=(
                        "The supplied evidence is insufficient to make a reliable "
                        "temporal comparison."
                    ),
                    citation_ids=[],
                    insufficient_evidence=True,
                ),
                evidence=evidence,
            )
        else:
            required_period_ids = tuple(
                set(item.evidence_id for item in period.evidence)
                for period in (comparison.older, comparison.newer)
                if period.evidence
            )
            comparison_block = (
                "Application temporal alignment result:\n"
                f"Topic: {comparison.topic.key}\n"
                f"Classification: {comparison.category.value}\n"
                "This category describes evidence alignment, not business materiality."
            )
            interpretation = self._generate(
                question=question,
                evidence=evidence,
                comparison_block=comparison_block,
                required_period_ids=required_period_ids,
            )
        return TemporalDisclosureAnalysis(
            question=question,
            comparison=comparison,
            interpretation=interpretation,
        )

    def _generate(
        self,
        *,
        question: str,
        evidence: tuple[NumberedEvidence, ...],
        comparison_block: str,
        required_period_ids: tuple[set[int], ...],
    ) -> GroundedGenerationResult:
        raw = self._llm_client.generate(
            system_prompt=TEMPORAL_SYSTEM_PROMPT,
            user_prompt=build_grounded_user_prompt(
                question,
                [item.result for item in evidence],
                deterministic_calculation=comparison_block,
            ),
            response_format=GroundedAnswer,
        )
        answer = parse_and_validate_answer(raw, evidence_count=len(evidence))
        if not answer.insufficient_evidence:
            returned = set(answer.citation_ids)
            if any(not returned.intersection(period_ids) for period_ids in required_period_ids):
                raise CitationValidationError(
                    "temporal answer must cite evidence from every available period"
                )
        return GroundedGenerationResult(answer=answer, evidence=evidence)
