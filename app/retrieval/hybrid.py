"""Hybrid vector/lexical retrieval, RRF fusion, and reranking."""

from dataclasses import replace

from sqlalchemy.orm import Session

from app.retrieval.lexical_store import LexicalStore
from app.retrieval.metadata import MetadataAwareRetriever, MetadataCatalog
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorSearchResult, VectorStore


def reciprocal_rank_fusion(
    vector_results: list[VectorSearchResult],
    lexical_results: list[VectorSearchResult],
    *,
    rrf_k: int = 60,
) -> list[VectorSearchResult]:
    """Fuse ranked channels without comparing their incompatible score scales."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    candidates: dict[int, VectorSearchResult] = {}
    contributions: dict[int, float] = {}
    vector_ranks: dict[int, int] = {}
    lexical_ranks: dict[int, int] = {}

    for rank, result in enumerate(vector_results, start=1):
        if result.chunk_id in vector_ranks:
            continue
        vector_ranks[result.chunk_id] = rank
        candidates[result.chunk_id] = result
        contributions[result.chunk_id] = 1.0 / (rrf_k + rank)

    for rank, result in enumerate(lexical_results, start=1):
        if result.chunk_id in lexical_ranks:
            continue
        lexical_ranks[result.chunk_id] = rank
        if result.chunk_id in candidates:
            _validate_same_chunk(candidates[result.chunk_id], result)
        else:
            candidates[result.chunk_id] = result
            contributions[result.chunk_id] = 0.0
        contributions[result.chunk_id] += 1.0 / (rrf_k + rank)

        existing = candidates[result.chunk_id]
        candidates[result.chunk_id] = replace(
            existing,
            lexical_rank=rank,
            lexical_score=result.lexical_score,
        )

    fused = [
        replace(
            result,
            vector_rank=vector_ranks.get(chunk_id),
            lexical_rank=lexical_ranks.get(chunk_id),
            fusion_score=contributions[chunk_id],
        )
        for chunk_id, result in candidates.items()
    ]
    fused.sort(
        key=lambda item: (
            -(item.fusion_score if item.fusion_score is not None else 0.0),
            min(item.vector_rank or 10**9, item.lexical_rank or 10**9),
            item.chunk_id,
        )
    )
    return fused


def _validate_same_chunk(
    left: VectorSearchResult, right: VectorSearchResult
) -> None:
    left_identity = (
        left.document_id,
        left.company,
        left.document_type,
        left.fiscal_year,
        left.page_number,
    )
    right_identity = (
        right.document_id,
        right.company,
        right.document_type,
        right.fiscal_year,
        right.page_number,
    )
    if left_identity != right_identity:
        raise ValueError("retrieval channels returned conflicting chunk provenance")


class HybridRetriever:
    """Run scoped vector and lexical channels, fuse, then rerank."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        reranker: Reranker,
        candidate_k: int = 20,
        rrf_k: int = 60,
        catalog: MetadataCatalog | None = None,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")
        self._vector_store = vector_store
        self._lexical_store = lexical_store
        self._reranker = reranker
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._metadata = MetadataAwareRetriever(vector_store, catalog=catalog)

    def search(
        self,
        db: Session,
        query: str,
        *,
        company: str | None = None,
        top_k: int = 5,
        document_id: int | None = None,
        document_type: str | None = None,
        fiscal_year: int | None = None,
    ) -> list[VectorSearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        plan = self._metadata.plan(
            db,
            query,
            company=company,
            document_id=document_id,
            document_type=document_type,
            fiscal_year=fiscal_year,
        )
        document = plan.document
        channel_k = max(self._candidate_k, top_k)
        scope = {
            "company": document.company,
            "top_k": channel_k,
            "document_id": document.document_id,
            "document_type": document.document_type,
            "fiscal_year": document.fiscal_year,
        }
        vector_results = self._vector_store.search(
            db,
            plan.semantic_query,
            **scope,
        )
        lexical_results = self._lexical_store.search(
            db,
            plan.semantic_query,
            **scope,
        )
        fused = reciprocal_rank_fusion(
            vector_results,
            lexical_results,
            rrf_k=self._rrf_k,
        )
        return self._reranker.rerank(
            plan.semantic_query,
            fused,
            top_k=top_k,
        )
