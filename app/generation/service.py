"""Orchestrate retrieval, grounded generation, and citation-ID validation."""

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from app.generation.llm_client import LLMClient
from app.generation.prompts import SYSTEM_PROMPT, build_grounded_user_prompt
from app.retrieval.base import Retriever
from app.retrieval.vector_store import VectorSearchResult


class StructuredResponseError(ValueError):
    """Raised when a provider does not return the required JSON structure."""


class CitationValidationError(ValueError):
    """Raised when an answer references evidence that was not supplied."""


class GroundedAnswer(BaseModel):
    """Structured answer contract shared with the LLM provider."""

    model_config = ConfigDict(extra="forbid", strict=True)

    answer: str = Field(
        min_length=1,
        description="Answer text with inline [n] markers after every supported claim.",
    )
    citation_ids: list[int] = Field(
        description="Unique evidence IDs used as inline [n] markers in answer."
    )
    insufficient_evidence: bool = Field(
        description="True only when the supplied evidence cannot answer the question."
    )


@dataclass(frozen=True)
class NumberedEvidence:
    """A retrieved result paired with its prompt-visible citation ID."""

    evidence_id: int
    result: VectorSearchResult


@dataclass(frozen=True)
class GroundedGenerationResult:
    """Validated answer plus the evidence needed to render citations later."""

    answer: GroundedAnswer
    evidence: tuple[NumberedEvidence, ...]


def parse_and_validate_answer(raw_response: str, *, evidence_count: int) -> GroundedAnswer:
    """Parse strict JSON and validate citation references, not entailment."""
    try:
        answer = GroundedAnswer.model_validate_json(raw_response)
    except ValidationError as exc:
        raise StructuredResponseError("LLM response did not match the answer schema") from exc

    if len(answer.citation_ids) != len(set(answer.citation_ids)):
        raise CitationValidationError("citation_ids must not contain duplicates")

    inline_ids = [int(value) for value in re.findall(r"\[(\d+)\]", answer.answer)]
    supplied_ids = set(range(1, evidence_count + 1))
    returned_ids = set(answer.citation_ids)
    referenced_ids = set(inline_ids)
    invalid_ids = (returned_ids | referenced_ids) - supplied_ids
    if invalid_ids:
        rendered = ", ".join(str(value) for value in sorted(invalid_ids))
        raise CitationValidationError(f"answer referenced unsupplied evidence IDs: {rendered}")

    if answer.insufficient_evidence:
        if returned_ids or referenced_ids:
            raise CitationValidationError(
                "insufficient-evidence answers must not include citations"
            )
        return answer

    if not returned_ids:
        raise CitationValidationError("a supported answer must include citations")
    if returned_ids != referenced_ids:
        raise CitationValidationError(
            "citation_ids must exactly match the evidence markers in the answer"
        )
    return answer


class GroundedGenerationService:
    """Retrieve scoped evidence and generate one validated grounded answer."""

    def __init__(self, *, retriever: Retriever, llm_client: LLMClient) -> None:
        self._retriever = retriever
        self._llm_client = llm_client

    def answer(
        self,
        db: Session,
        question: str,
        *,
        company: str | None = None,
        top_k: int = 5,
        document_id: int | None = None,
        document_type: str | None = None,
        fiscal_year: int | None = None,
    ) -> GroundedGenerationResult:
        results = self._retriever.search(
            db,
            question,
            company=company,
            top_k=top_k,
            document_id=document_id,
            document_type=document_type,
            fiscal_year=fiscal_year,
        )
        evidence = tuple(
            NumberedEvidence(evidence_id=index, result=result)
            for index, result in enumerate(results, start=1)
        )

        if not evidence:
            return GroundedGenerationResult(
                answer=GroundedAnswer(
                    answer="The supplied evidence is insufficient to answer the question.",
                    citation_ids=[],
                    insufficient_evidence=True,
                ),
                evidence=evidence,
            )

        raw_response = self._llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_grounded_user_prompt(question, results),
            response_format=GroundedAnswer,
        )
        answer = parse_and_validate_answer(raw_response, evidence_count=len(evidence))
        return GroundedGenerationResult(answer=answer, evidence=evidence)
