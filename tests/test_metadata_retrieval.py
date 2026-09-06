"""Deterministic tests for M5 metadata-aware query routing."""

from types import SimpleNamespace

import pytest

from app.generation.service import GroundedGenerationService
from app.retrieval.metadata import (
    AmbiguousMetadataError,
    DocumentMetadata,
    MetadataAwareRetriever,
    MetadataConflictError,
    MissingMetadataScopeError,
    MissingSemanticQueryError,
    UnsupportedMetadataError,
    UnsupportedTemporalRequestError,
    build_metadata_query_plan,
)


APPLE_2025 = DocumentMetadata(1, "Apple", "10-K", 2025)


def test_query_metadata_resolves_catalog_document_and_cleans_semantic_query() -> None:
    plan = build_metadata_query_plan(
        "What supply-chain risks did Apple identify in its 2025 10-K?",
        [APPLE_2025],
    )
    assert plan.document == APPLE_2025
    assert plan.semantic_query == "What supply-chain risks did Apple identify?"


def test_annual_report_alias_maps_to_available_10_k() -> None:
    plan = build_metadata_query_plan(
        "What risks did Apple disclose from the 2025 annual report?",
        [APPLE_2025],
    )
    assert plan.document.document_type == "10-K"
    assert plan.semantic_query == "What risks did Apple disclose?"


def test_according_to_scope_phrase_is_removed_as_a_unit() -> None:
    plan = build_metadata_query_plan(
        "According to the 2025 10-K, what affected Apple's net sales?",
        [APPLE_2025],
    )
    assert plan.semantic_query == "what affected Apple's net sales?"


def test_document_type_before_year_scope_phrase_is_removed_as_a_unit() -> None:
    plan = build_metadata_query_plan(
        "What risks did Apple disclose in its 10-K for 2025?",
        [APPLE_2025],
    )
    assert plan.semantic_query == "What risks did Apple disclose?"


def test_metadata_outside_a_complete_scope_phrase_is_retained() -> None:
    query = "What risks are in Apple's 2025 annual report?"
    plan = build_metadata_query_plan(query, [APPLE_2025])
    assert plan.semantic_query == query


def test_explicit_document_id_in_query_resolves_catalog_document() -> None:
    plan = build_metadata_query_plan(
        "What risks are disclosed in document id 1?",
        [APPLE_2025],
    )
    assert plan.document == APPLE_2025
    assert plan.semantic_query == "What risks are disclosed?"


def test_multiple_supported_companies_are_ambiguous() -> None:
    with pytest.raises(AmbiguousMetadataError, match="multiple supported companies"):
        build_metadata_query_plan(
            "Compare Apple and Microsoft risks in the 2025 10-K.",
            [APPLE_2025, DocumentMetadata(2, "Microsoft", "10-K", 2025)],
        )


def test_explicit_metadata_can_complete_query_scope() -> None:
    plan = build_metadata_query_plan(
        "What supply-chain risks were reported?",
        [APPLE_2025],
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
    )
    assert plan.document == APPLE_2025
    assert plan.semantic_query == "What supply-chain risks were reported?"


def test_query_and_explicit_metadata_conflict_is_rejected() -> None:
    with pytest.raises(MetadataConflictError, match="fiscal year"):
        build_metadata_query_plan(
            "What did Apple report in 2025?",
            [APPLE_2025],
            fiscal_year=2024,
        )


def test_multiple_years_are_reserved_for_temporal_milestone() -> None:
    with pytest.raises(UnsupportedTemporalRequestError, match="multi-year"):
        build_metadata_query_plan(
            "Compare Apple revenue in 2024 and 2025.",
            [APPLE_2025],
        )


def test_unavailable_document_scope_is_rejected() -> None:
    with pytest.raises(UnsupportedMetadataError, match="not available"):
        build_metadata_query_plan(
            "What did Apple report in its 2024 10-K?",
            [APPLE_2025],
        )


def test_underspecified_scope_with_multiple_documents_is_ambiguous() -> None:
    with pytest.raises(AmbiguousMetadataError, match="multiple documents"):
        build_metadata_query_plan(
            "What risks did Apple report?",
            [APPLE_2025, DocumentMetadata(2, "Apple", "10-Q", 2025)],
        )


def test_company_scope_is_required_instead_of_silently_guessing() -> None:
    with pytest.raises(MissingMetadataScopeError, match="company"):
        build_metadata_query_plan("What were the major risks?", [APPLE_2025])


def test_metadata_only_query_has_no_semantic_search_subject() -> None:
    with pytest.raises(MissingSemanticQueryError, match="subject"):
        build_metadata_query_plan("Apple 2025 10-K", [APPLE_2025])


class FakeCatalog:
    def list_documents(self, db: object) -> list[DocumentMetadata]:
        return [APPLE_2025]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str, dict[str, object]]] = []

    def search(self, db: object, query: str, **kwargs: object) -> list[object]:
        self.calls.append((db, query, kwargs))
        return []


def test_metadata_retriever_delegates_clean_query_and_resolved_filters() -> None:
    vector_store = FakeVectorStore()
    retriever = MetadataAwareRetriever(
        vector_store,  # type: ignore[arg-type]
        catalog=FakeCatalog(),  # type: ignore[arg-type]
    )
    db = object()
    retriever.search(
        db,  # type: ignore[arg-type]
        "What risks did Apple identify in the 2025 10-K?",
        top_k=7,
    )
    assert vector_store.calls == [
        (
            db,
            "What risks did Apple identify?",
            {
                "company": "Apple",
                "top_k": 7,
                "document_id": 1,
                "document_type": "10-K",
                "fiscal_year": 2025,
            },
        )
    ]


class FakeLLM:
    def generate(self, **kwargs: object) -> str:
        assert "Apple" in str(kwargs["user_prompt"])
        return (
            '{"answer":"The filing identifies supply risks [1].",'
            '"citation_ids":[1],"insufficient_evidence":false}'
        )


class ResultVectorStore(FakeVectorStore):
    def search(self, db: object, query: str, **kwargs: object) -> list[object]:
        self.calls.append((db, query, kwargs))
        return [
            SimpleNamespace(
                chunk_id=45,
                text="The company may experience supply shortages.",
                page_number=9,
                document_id=1,
                company="Apple",
                document_type="10-K",
                fiscal_year=2025,
                cosine_distance=0.2,
                similarity=0.8,
                lexical_score=None,
                fusion_score=None,
                rerank_score=None,
            )
        ]


def test_metadata_aware_retriever_integrates_with_grounded_generation() -> None:
    vector_store = ResultVectorStore()
    retriever = MetadataAwareRetriever(
        vector_store,  # type: ignore[arg-type]
        catalog=FakeCatalog(),  # type: ignore[arg-type]
    )
    service = GroundedGenerationService(retriever=retriever, llm_client=FakeLLM())
    result = service.answer(
        object(),  # type: ignore[arg-type]
        "What supply risks did Apple identify in its 2025 10-K?",
    )
    assert result.answer.citation_ids == [1]
    assert result.evidence[0].result.company == "Apple"
    assert result.evidence[0].result.fiscal_year == 2025
