"""Integration tests for revenue growth orchestration with fakes."""

import json
from decimal import Decimal

from app.financial.service import RevenueGrowthService
from app.retrieval.vector_store import VectorSearchResult


class FakeRetriever:
    def __init__(self, result: VectorSearchResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    def search(self, db: object, query: str, **kwargs: object) -> list[VectorSearchResult]:
        self.calls.append((query, kwargs))
        return [self.result]


class FakeLLM:
    def __init__(self) -> None:
        self.user_prompt = ""

    def generate(self, **kwargs: object) -> str:
        self.user_prompt = str(kwargs["user_prompt"])
        return json.dumps(
            {
                "answer": "Total net sales grew 6.4% from 2024 to 2025 [1].",
                "citation_ids": [1],
                "insufficient_evidence": False,
            }
        )


def test_service_keeps_calculation_separate_from_generated_interpretation() -> None:
    result = VectorSearchResult(
        chunk_id=141,
        text=(
            "2025 Change 2024 Change 2023 (dollars in millions): "
            "Total net sales $ 416,161 6 %$ 391,035 2 %$ 383,285"
        ),
        page_number=26,
        document_id=1,
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        cosine_distance=0.1,
        similarity=0.9,
    )
    retriever = FakeRetriever(result)
    llm = FakeLLM()
    service = RevenueGrowthService(
        retriever=retriever,  # type: ignore[arg-type]
        llm_client=llm,
    )

    analysis = service.analyze(
        object(),  # type: ignore[arg-type]
        company="Apple",
        current_year=2025,
        prior_year=2024,
    )

    assert analysis.calculation.percentage_change.as_tuple() == Decimal("6.4").as_tuple()
    assert analysis.calculation.current.chunk_id == 141
    assert analysis.interpretation.answer.citation_ids == [1]
    assert "Calculated result: +6.4%" in llm.user_prompt
    assert "authoritative; do not recompute" in llm.user_prompt
    assert retriever.calls == [
        (
            "total net sales revenue 2025 2024",
            {
                "company": "Apple",
                "top_k": 5,
                "document_type": "10-K",
                "fiscal_year": 2025,
            },
        )
    ]
