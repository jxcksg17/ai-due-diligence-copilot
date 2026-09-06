"""Unit tests for local cross-encoder reranking without model downloads."""

import pytest

from app.config import Settings
from app.retrieval.reranker import CrossEncoderReranker, get_reranker
from app.retrieval.vector_store import VectorSearchResult


def _candidate(chunk_id: int, fusion_score: float) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=f"passage {chunk_id}",
        page_number=chunk_id,
        document_id=1,
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        cosine_distance=0.2,
        similarity=0.8,
        fusion_score=fusion_score,
    )


class FakeEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[object, ...]] = []

    def predict(self, pairs: object, **kwargs: object) -> list[float]:
        self.calls.append((pairs, kwargs))
        return self.scores


def test_cross_encoder_reranks_after_fusion_and_preserves_scores() -> None:
    encoder = FakeEncoder([0.1, 0.9, 0.5])
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size=8,
        device="cpu",
        encoder=encoder,
    )
    results = reranker.rerank(
        "revenue risk",
        [_candidate(1, 0.03), _candidate(2, 0.02), _candidate(3, 0.01)],
        top_k=2,
    )
    assert [item.chunk_id for item in results] == [2, 3]
    assert [item.rerank_score for item in results] == [0.9, 0.5]
    assert results[0].fusion_score == 0.02
    pairs, kwargs = encoder.calls[0]
    assert pairs == [
        ("revenue risk", "passage 1"),
        ("revenue risk", "passage 2"),
        ("revenue risk", "passage 3"),
    ]
    assert kwargs == {
        "batch_size": 8,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }


def test_reranker_factory_uses_configured_lightweight_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    class FakeReranker:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)

    monkeypatch.setattr("app.retrieval.reranker.CrossEncoderReranker", FakeReranker)
    settings = Settings(database_url="sqlite://")
    get_reranker(settings)
    assert created == {
        "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "batch_size": 8,
    }
