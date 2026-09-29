"""Real M6 retrieval evaluation with stage-level latency accounting."""

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.embeddings.client import EmbeddingClient
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.lexical_store import LexicalStore
from app.retrieval.metadata import MetadataAwareRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorSearchResult, VectorStore
from evals.metrics import mean
from evals.schema import EvaluationCase, EvidenceTarget


@dataclass
class TimedEmbeddingClient:
    """Measure query encoding without changing the application provider."""

    wrapped: EmbeddingClient
    last_query_seconds: float = 0.0

    def embed(self, text: str) -> list[float]:
        return self.wrapped.embed(text)

    def embed_query(self, text: str) -> list[float]:
        started = perf_counter()
        result = self.wrapped.embed_query(text)
        self.last_query_seconds = perf_counter() - started
        return result


@dataclass(frozen=True)
class RetrievalArtifacts:
    metrics: dict[str, Any]
    latencies: dict[str, Any]
    final_results: dict[str, list[VectorSearchResult]]


def evaluate_retrieval(
    db: Session,
    cases: list[EvaluationCase],
    *,
    embedding_client: EmbeddingClient,
    reranker: Reranker,
    candidate_k: int,
    rrf_k: int,
    top_k: int = 5,
) -> RetrievalArtifacts:
    if not cases:
        raise ValueError("retrieval evaluation requires at least one case")
    timed_embedding = TimedEmbeddingClient(embedding_client)
    vector_store = VectorStore(timed_embedding)
    lexical_store = LexicalStore()
    metadata = MetadataAwareRetriever(vector_store)
    channel_scores: dict[str, dict[str, list[float]]] = {
        channel: defaultdict(list)
        for channel in ("vector_only", "hybrid", "hybrid_rerank")
    }
    timings: dict[str, list[float]] = defaultdict(list)
    final_results: dict[str, list[VectorSearchResult]] = {}

    for case in cases:
        if case.scope is None or case.question is None:
            raise ValueError(f"retrieval case {case.id} is missing scope or question")
        scope = case.scope
        started = perf_counter()
        plan = metadata.plan(
            db,
            case.question,
            company=scope.company,
            document_type=scope.document_type,
            fiscal_year=scope.fiscal_year,
        )
        timings["metadata_resolution"].append(perf_counter() - started)
        query = plan.semantic_query
        search_scope = {
            "company": plan.document.company,
            "top_k": max(candidate_k, top_k),
            "document_id": plan.document.document_id,
            "document_type": plan.document.document_type,
            "fiscal_year": plan.document.fiscal_year,
        }

        started = perf_counter()
        dense = vector_store.search(db, query, **search_scope)
        dense_total = perf_counter() - started
        timings["query_encoding"].append(timed_embedding.last_query_seconds)
        timings["dense_retrieval"].append(
            max(0.0, dense_total - timed_embedding.last_query_seconds)
        )

        started = perf_counter()
        lexical = lexical_store.search(db, query, **search_scope)
        timings["lexical_retrieval"].append(perf_counter() - started)

        started = perf_counter()
        fused = reciprocal_rank_fusion(dense, lexical, rrf_k=rrf_k)
        timings["fusion"].append(perf_counter() - started)

        started = perf_counter()
        reranked = reranker.rerank(query, fused, top_k=top_k)
        timings["reranking"].append(perf_counter() - started)
        final_results[case.id] = reranked

        for channel, results in (
            ("vector_only", dense[:top_k]),
            ("hybrid", fused[:top_k]),
            ("hybrid_rerank", reranked),
        ):
            scores = _case_metrics(results, case.expected_evidence, ks=(1, 3, 5))
            for metric, value in scores.items():
                channel_scores[channel][metric].append(value)

    metrics: dict[str, Any] = {}
    for channel, scores in channel_scores.items():
        metrics[channel] = {
            metric: round(mean(values), 6) for metric, values in sorted(scores.items())
        }
        metrics[channel]["case_count"] = len(cases)
    latency_summary = {
        stage: _latency_summary(values) for stage, values in sorted(timings.items())
    }
    return RetrievalArtifacts(metrics, latency_summary, final_results)


def evidence_target_matches(result: VectorSearchResult, target: EvidenceTarget) -> bool:
    if result.page_number != target.page:
        return False
    normalized = _normalize(result.text)
    return all(_normalize(anchor) in normalized for anchor in target.anchors)


def _case_metrics(
    results: list[VectorSearchResult],
    targets: list[EvidenceTarget],
    *,
    ks: tuple[int, ...],
) -> dict[str, float]:
    target_hits_by_rank = [
        {
            index
            for index, target in enumerate(targets)
            if evidence_target_matches(result, target)
        }
        for result in results
    ]
    output: dict[str, float] = {}
    for k in ks:
        found = set().union(*target_hits_by_rank[:k]) if target_hits_by_rank[:k] else set()
        relevant_results = sum(bool(hits) for hits in target_hits_by_rank[:k])
        output[f"recall_at_{k}"] = len(found) / len(targets)
        output[f"precision_at_{k}"] = relevant_results / k
    output["mrr"] = next(
        (1.0 / rank for rank, hits in enumerate(target_hits_by_rank, start=1) if hits),
        0.0,
    )
    return output


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    summary = {
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "max": round(max(values), 6),
        "cold_start": round(values[0], 6),
    }
    if len(values) > 1:
        summary["warm_mean"] = round(mean(values[1:]), 6)
    return summary


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
