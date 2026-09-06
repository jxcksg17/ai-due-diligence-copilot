"""Local cross-encoder reranking behind a small provider interface."""

from dataclasses import replace
from typing import Any, Protocol

from app.config import Settings
from app.retrieval.vector_store import VectorSearchResult


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Reranker(Protocol):
    """Rerank fused candidates using the query and full chunk text."""

    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Return at most top_k candidates ordered by reranker score."""


def _select_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker:
    """Sentence Transformers cross-encoder with MPS and CPU fallback."""

    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        device: str | None = None,
        encoder: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._batch_size = batch_size
        self.device = device or _select_device()
        if encoder is not None:
            self._encoder = encoder
            return

        from sentence_transformers import CrossEncoder

        try:
            self._encoder = CrossEncoder(model_name, device=self.device, max_length=512)
        except RuntimeError:
            if self.device != "mps":
                raise
            self.device = "cpu"
            self._encoder = CrossEncoder(model_name, device="cpu", max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self._encoder.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scored = [
            replace(candidate, rerank_score=float(score))
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(
            key=lambda item: (
                -(item.rerank_score if item.rerank_score is not None else float("-inf")),
                -(item.fusion_score if item.fusion_score is not None else 0.0),
                item.chunk_id,
            )
        )
        return scored[:top_k]


def get_reranker(settings: Settings) -> Reranker:
    if settings.reranker_model != RERANKER_MODEL:
        raise ValueError(
            f"Unsupported RERANKER_MODEL {settings.reranker_model!r}; "
            f"expected {RERANKER_MODEL!r}"
        )
    return CrossEncoderReranker(
        model_name=settings.reranker_model,
        batch_size=settings.reranker_batch_size,
    )
