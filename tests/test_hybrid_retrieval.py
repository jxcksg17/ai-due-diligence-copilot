"""Tests for RRF fusion and scoped hybrid orchestration."""

from app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult


def _result(
    chunk_id: int,
    *,
    similarity: float | None = None,
    lexical_score: float | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=f"chunk {chunk_id}",
        page_number=chunk_id,
        document_id=1,
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        cosine_distance=None if similarity is None else 1.0 - similarity,
        similarity=similarity,
        lexical_score=lexical_score,
    )


def test_rrf_rewards_candidates_found_by_both_channels() -> None:
    fused = reciprocal_rank_fusion(
        [_result(1, similarity=0.9), _result(2, similarity=0.8)],
        [_result(2, lexical_score=0.7), _result(3, lexical_score=0.6)],
        rrf_k=60,
    )
    assert [item.chunk_id for item in fused] == [2, 1, 3]
    assert fused[0].vector_rank == 2
    assert fused[0].lexical_rank == 1
    assert fused[0].similarity == 0.8
    assert fused[0].lexical_score == 0.7


def test_rrf_deduplicates_repeated_candidates_within_channels() -> None:
    fused = reciprocal_rank_fusion(
        [_result(1, similarity=0.9), _result(1, similarity=0.8)],
        [_result(1, lexical_score=0.5), _result(1, lexical_score=0.4)],
    )
    assert len(fused) == 1
    assert fused[0].vector_rank == 1
    assert fused[0].lexical_rank == 1


def test_rrf_falls_back_when_lexical_channel_is_empty() -> None:
    fused = reciprocal_rank_fusion(
        [_result(1, similarity=0.9), _result(2, similarity=0.8)],
        [],
    )
    assert [item.chunk_id for item in fused] == [1, 2]
    assert all(item.lexical_rank is None for item in fused)


def test_rrf_falls_back_when_vector_channel_is_empty() -> None:
    fused = reciprocal_rank_fusion(
        [],
        [_result(2, lexical_score=0.7), _result(3, lexical_score=0.6)],
    )
    assert [item.chunk_id for item in fused] == [2, 3]
    assert all(item.vector_rank is None for item in fused)


class FakeCatalog:
    def list_documents(self, db: object) -> list[DocumentMetadata]:
        return [DocumentMetadata(1, "Apple", "10-K", 2025)]


class RecordingStore:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, object]]] = []

    def search(self, db: object, query: str, **kwargs: object) -> list[VectorSearchResult]:
        self.calls.append((query, kwargs))
        return self.results


class ReverseReranker:
    def __init__(self) -> None:
        self.candidates: list[VectorSearchResult] = []

    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        self.candidates = candidates
        return list(reversed(candidates))[:top_k]


def test_hybrid_retriever_scopes_both_channels_then_reranks_fused_candidates() -> None:
    vector = RecordingStore([_result(1, similarity=0.9), _result(2, similarity=0.8)])
    lexical = RecordingStore([_result(2, lexical_score=0.7), _result(3, lexical_score=0.6)])
    reranker = ReverseReranker()
    retriever = HybridRetriever(
        vector_store=vector,  # type: ignore[arg-type]
        lexical_store=lexical,  # type: ignore[arg-type]
        reranker=reranker,
        candidate_k=10,
        catalog=FakeCatalog(),  # type: ignore[arg-type]
    )
    results = retriever.search(
        object(),  # type: ignore[arg-type]
        "What risks did Apple identify in its 2025 10-K?",
        top_k=2,
    )
    expected_scope = {
        "company": "Apple",
        "top_k": 10,
        "document_id": 1,
        "document_type": "10-K",
        "fiscal_year": 2025,
    }
    assert vector.calls == [("What risks did Apple identify?", expected_scope)]
    assert lexical.calls == [("What risks did Apple identify?", expected_scope)]
    assert [item.chunk_id for item in reranker.candidates] == [2, 1, 3]
    assert [item.chunk_id for item in results] == [3, 1]


def test_hybrid_preserves_vector_candidates_when_lexical_has_no_results() -> None:
    vector = RecordingStore([_result(1, similarity=0.9)])
    lexical = RecordingStore([])
    retriever = HybridRetriever(
        vector_store=vector,  # type: ignore[arg-type]
        lexical_store=lexical,  # type: ignore[arg-type]
        reranker=ReverseReranker(),
        catalog=FakeCatalog(),  # type: ignore[arg-type]
    )
    results = retriever.search(
        object(),  # type: ignore[arg-type]
        "What risks did Apple identify?",
    )
    assert results[0].chunk_id == 1
    assert results[0].similarity == 0.9
