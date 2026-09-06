"""M4 orchestration for deterministic revenue-growth analysis."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.financial.revenue_growth import (
    RevenueGrowthResult,
    calculate_revenue_growth,
    extract_revenue_growth_inputs,
)
from app.generation.llm_client import LLMClient
from app.generation.prompts import SYSTEM_PROMPT, build_grounded_user_prompt
from app.generation.service import (
    GroundedAnswer,
    GroundedGenerationResult,
    NumberedEvidence,
    parse_and_validate_answer,
)
from app.retrieval.vector_store import VectorStore


@dataclass(frozen=True)
class RevenueGrowthAnalysis:
    """Source-backed calculation and its generated interpretation."""

    calculation: RevenueGrowthResult
    interpretation: GroundedGenerationResult


class RevenueGrowthService:
    """Retrieve explicit revenue values, calculate growth, then explain it."""

    def __init__(self, *, retriever: VectorStore, llm_client: LLMClient) -> None:
        self._retriever = retriever
        self._llm_client = llm_client

    def analyze(
        self,
        db: Session,
        *,
        company: str,
        current_year: int,
        prior_year: int,
        document_type: str = "10-K",
        top_k: int = 5,
        rounding_places: int = 1,
    ) -> RevenueGrowthAnalysis:
        query = f"total net sales revenue {current_year} {prior_year}"
        results = self._retriever.search(
            db,
            query,
            company=company,
            top_k=top_k,
            document_type=document_type,
            fiscal_year=current_year,
        )
        current, prior = extract_revenue_growth_inputs(
            results,
            current_year=current_year,
            prior_year=prior_year,
        )
        calculation = calculate_revenue_growth(
            current,
            prior,
            rounding_places=rounding_places,
        )
        question = (
            f"How did {company}'s total net sales change from "
            f"{prior_year} to {current_year}?"
        )
        raw_response = self._llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_grounded_user_prompt(
                question,
                results,
                deterministic_calculation=calculation.as_prompt_block(),
            ),
            response_format=GroundedAnswer,
        )
        answer = parse_and_validate_answer(raw_response, evidence_count=len(results))
        evidence = tuple(
            NumberedEvidence(evidence_id=index, result=result)
            for index, result in enumerate(results, start=1)
        )
        return RevenueGrowthAnalysis(
            calculation=calculation,
            interpretation=GroundedGenerationResult(answer=answer, evidence=evidence),
        )
