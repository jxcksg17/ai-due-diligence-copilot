"""Grounded Risk Radar interpretation over deterministic filing signals."""

import re
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
from app.risk_radar.models import RiskComparison, RiskTemporalState


RISK_RADAR_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are interpreting an application-produced Risk Radar signal. Clearly label
your response as interpretation and use exactly one concise sentence. Place all
inline evidence markers together at the end of that sentence. Do not introduce
a risk outside the supplied topic or evidence. Do not assign
severity, probability, confidence, or financial impact. Do not claim that an
event will occur. Never override the deterministic presence or temporal signal.
Apply the temporal state as a constraint, but do not repeat its internal label in
the interpretation because it is displayed separately from source-backed prose.
When temporal alignment is ambiguous, summarize what each filing discloses
without asserting that the risk increased, decreased, or materially changed.
"""


@dataclass(frozen=True)
class RiskRadarAnalysis:
    question: str
    comparison: RiskComparison
    interpretation: GroundedGenerationResult


def _signal_block(comparison: RiskComparison) -> str:
    older_ids = ", ".join(
        f"[{item.evidence_id}]" for item in comparison.older.evidence
    ) or "none"
    newer_ids = ", ".join(
        f"[{item.evidence_id}]" for item in comparison.newer.evidence
    ) or "none"
    return (
        "Deterministic Risk Radar signal (authoritative; do not override):\n"
        f"Topic: {comparison.topic.label} ({comparison.topic.key})\n"
        f"Older filing ({comparison.older.document.fiscal_year}): "
        f"{comparison.older.presence.value}\n"
        f"Newer filing ({comparison.newer.document.fiscal_year}): "
        f"{comparison.newer.presence.value}\n"
        f"Temporal state: {comparison.temporal_state.value}\n"
        f"Older-period evidence IDs: {older_ids}\n"
        f"Newer-period evidence IDs: {newer_ids}\n"
        "A supported temporal interpretation must cite at least one supplied "
        "evidence ID from every disclosed period. Return exactly one concise "
        "sentence and place all citation markers at its end, with no uncited "
        "text after them. Do not repeat the internal temporal-state label in "
        "the interpretation.\n"
        "A disclosed state means configured anchors matched explicit Item 1A "
        "text. A not-found state means no configured anchor match was found; it "
        "does not prove the real-world risk is absent. Recurring-ambiguous means "
        "both filings disclosed the topic but semantic change was not established."
    )


class RiskRadarGenerationService:
    """Interpret a Risk Radar entry without blending evidence and signal layers."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def interpret(
        self,
        *,
        question: str,
        comparison: RiskComparison,
    ) -> RiskRadarAnalysis:
        if not question.strip():
            raise ValueError("question must not be empty")
        risk_evidence = (*comparison.older.evidence, *comparison.newer.evidence)
        evidence = tuple(
            NumberedEvidence(item.evidence_id, item.result)
            for item in risk_evidence
        )
        if comparison.temporal_state in {
            RiskTemporalState.NOT_FOUND_BOTH,
            RiskTemporalState.INSUFFICIENT,
        } or not evidence:
            interpretation = GroundedGenerationResult(
                answer=GroundedAnswer(
                    answer=(
                        "Interpretation: The supplied Risk Radar evidence is "
                        "insufficient for this topic."
                    ),
                    citation_ids=[],
                    insufficient_evidence=True,
                ),
                evidence=evidence,
            )
            return RiskRadarAnalysis(question, comparison, interpretation)

        raw = self._llm_client.generate(
            system_prompt=RISK_RADAR_SYSTEM_PROMPT,
            user_prompt=build_grounded_user_prompt(
                question,
                [item.result for item in evidence],
                deterministic_calculation=_signal_block(comparison),
            ),
            response_format=GroundedAnswer,
        )
        answer = parse_and_validate_answer(raw, evidence_count=len(evidence))
        if not answer.insufficient_evidence:
            _reject_uncited_tail(answer.answer)
            returned = set(answer.citation_ids)
            required_period_ids = [
                {item.evidence_id for item in signal.evidence}
                for signal in (comparison.older, comparison.newer)
                if signal.evidence
            ]
            if any(
                not returned.intersection(period_ids)
                for period_ids in required_period_ids
            ):
                raise CitationValidationError(
                    "Risk Radar interpretation must cite every disclosed period"
                )
        return RiskRadarAnalysis(
            question=question,
            comparison=comparison,
            interpretation=GroundedGenerationResult(answer, evidence),
        )


def _reject_uncited_tail(answer: str) -> None:
    markers = list(re.finditer(r"\[\d+\]", answer))
    if not markers:
        raise CitationValidationError("Risk Radar interpretation requires citations")
    tail = answer[markers[-1].end() :].strip(" \t\r\n.,;:!?")
    if tail:
        raise CitationValidationError(
            "Risk Radar interpretation contains uncited text after its final citation"
        )
