"""Tests for grounded generation orchestration and citation validation."""

import json

import pytest

from app.generation.service import (
    CitationValidationError,
    GroundedGenerationService,
    StructuredResponseError,
    parse_and_validate_answer,
)
from app.retrieval.vector_store import VectorSearchResult


def _result(chunk_id: int = 17, page: int = 29) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text="Net sales increased year over year.",
        page_number=page,
        document_id=1,
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        cosine_distance=0.18,
        similarity=0.82,
    )


class FakeRetriever:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[object, str, dict[str, object]]] = []

    def search(self, db: object, query: str, **kwargs: object) -> list[VectorSearchResult]:
        self.calls.append((db, query, kwargs))
        return self.results


class FakeLLMClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.response)


def test_service_integrates_scoped_retrieval_and_validated_generation() -> None:
    retriever = FakeRetriever([_result()])
    llm = FakeLLMClient(
        {
            "answer": "Net sales increased year over year [1].",
            "citation_ids": [1],
            "insufficient_evidence": False,
        }
    )
    service = GroundedGenerationService(
        retriever=retriever,  # type: ignore[arg-type]
        llm_client=llm,
    )

    result = service.answer(
        object(),  # type: ignore[arg-type]
        "How did net sales change?",
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        top_k=3,
    )

    assert result.answer.citation_ids == [1]
    assert result.evidence[0].evidence_id == 1
    assert result.evidence[0].result.page_number == 29
    assert retriever.calls[0][2] == {
        "company": "Apple",
        "top_k": 3,
        "document_id": None,
        "document_type": "10-K",
        "fiscal_year": 2025,
    }
    assert "[1]" in str(llm.calls[0]["user_prompt"])


def test_no_retrieved_evidence_returns_insufficient_without_calling_llm() -> None:
    llm = FakeLLMClient({})
    service = GroundedGenerationService(
        retriever=FakeRetriever([]),  # type: ignore[arg-type]
        llm_client=llm,
    )

    result = service.answer(object(), "Unknown fact?", company="Apple")  # type: ignore[arg-type]

    assert result.answer.insufficient_evidence is True
    assert result.answer.citation_ids == []
    assert result.evidence == ()
    assert llm.calls == []


def test_structured_insufficient_evidence_response_is_accepted() -> None:
    answer = parse_and_validate_answer(
        json.dumps(
            {
                "answer": "The supplied evidence does not answer the question.",
                "citation_ids": [],
                "insufficient_evidence": True,
            }
        ),
        evidence_count=2,
    )
    assert answer.insufficient_evidence is True


@pytest.mark.parametrize(
    "payload",
    [
        {
            "answer": "Unsupported claim [3].",
            "citation_ids": [3],
            "insufficient_evidence": False,
        },
        {
            "answer": "Claim cites one block [1].",
            "citation_ids": [2],
            "insufficient_evidence": False,
        },
        {
            "answer": "Claim without a marker.",
            "citation_ids": [],
            "insufficient_evidence": False,
        },
        {
            "answer": "Not enough evidence [1].",
            "citation_ids": [1],
            "insufficient_evidence": True,
        },
    ],
)
def test_invalid_or_inconsistent_citation_ids_are_rejected(
    payload: dict[str, object],
) -> None:
    with pytest.raises(CitationValidationError):
        parse_and_validate_answer(json.dumps(payload), evidence_count=2)


def test_malformed_structured_response_is_rejected() -> None:
    with pytest.raises(StructuredResponseError):
        parse_and_validate_answer("not json", evidence_count=1)
