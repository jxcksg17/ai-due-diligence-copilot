"""Tests for deterministic evidence prompt construction."""

from app.generation.prompts import SYSTEM_PROMPT, build_grounded_user_prompt
from app.retrieval.vector_store import VectorSearchResult


def _result(chunk_id: int, page: int, text: str) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        page_number=page,
        document_id=1,
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        cosine_distance=0.2,
        similarity=0.8,
    )


def test_prompt_numbers_evidence_and_includes_citation_metadata() -> None:
    prompt = build_grounded_user_prompt(
        "What affected sales?",
        [_result(10, 20, "First passage"), _result(11, 21, "Second passage")],
    )

    assert prompt.index("[1]") < prompt.index("[2]")
    assert "chunk_id=10; document_id=1; company=Apple" in prompt
    assert "document_type=10-K; fiscal_year=2025; page=20" in prompt
    assert "First passage" in prompt
    assert "Second passage" in prompt


def test_system_prompt_contains_grounding_and_financial_calculation_guards() -> None:
    assert "Use only the evidence blocks" in SYSTEM_PROMPT
    assert "new derived financial calculation as authoritative" in SYSTEM_PROMPT
    assert "insufficient_evidence to true" in SYSTEM_PROMPT
    assert "including short one-sentence answers" in SYSTEM_PROMPT
    assert '"citation_ids":[1]' in SYSTEM_PROMPT


def test_prompt_preserves_hybrid_score_provenance() -> None:
    result = _result(10, 20, "Hybrid evidence")
    result = VectorSearchResult(
        **{
            **result.__dict__,
            "lexical_rank": 2,
            "lexical_score": 0.4,
            "fusion_score": 0.03,
            "rerank_score": 2.5,
        }
    )
    prompt = build_grounded_user_prompt("Question?", [result])
    assert "vector_similarity=0.800000" in prompt
    assert "lexical_score=0.400000" in prompt
    assert "fusion_score=0.030000" in prompt
    assert "rerank_score=2.500000" in prompt
