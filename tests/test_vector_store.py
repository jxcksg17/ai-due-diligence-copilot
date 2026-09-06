"""Deterministic tests for exact pgvector retrieval orchestration."""

from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from app.retrieval.vector_store import VectorStore


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, text: str) -> list[float]:
        raise AssertionError("retrieval must not use document embedding")

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1] * 1024


class FakeResult:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def all(self) -> list[SimpleNamespace]:
        return self._rows


class FakeSession:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self.rows = rows
        self.statement = None

    def execute(self, statement: object) -> FakeResult:
        self.statement = statement
        return FakeResult(self.rows)


def test_search_uses_query_embedding_and_maps_citation_metadata() -> None:
    client = FakeEmbeddingClient()
    db = FakeSession(
        [
            SimpleNamespace(
                chunk_id=17,
                text="Net sales increased year over year.",
                page_number=29,
                document_id=1,
                company="Apple",
                document_type="10-K",
                fiscal_year=2025,
                cosine_distance=0.18,
            )
        ]
    )

    results = VectorStore(client).search(
        db,  # type: ignore[arg-type]
        "How did net sales change?",
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        top_k=3,
    )

    assert client.queries == ["How did net sales change?"]
    assert len(results) == 1
    assert results[0].chunk_id == 17
    assert results[0].page_number == 29
    assert results[0].company == "Apple"
    assert results[0].document_type == "10-K"
    assert results[0].fiscal_year == 2025
    assert results[0].cosine_distance == pytest.approx(0.18)
    assert results[0].similarity == pytest.approx(0.82)


def test_statement_uses_cosine_distance_top_k_and_explicit_filters() -> None:
    statement = VectorStore._build_statement(
        [0.1] * 1024,
        company="Apple",
        top_k=7,
        document_id=1,
        document_type="10-K",
        fiscal_year=2025,
    )
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert "chunks.embedding <=>" in sql
    assert "chunks.embedding IS NOT NULL" in sql
    assert "companies.name =" in sql
    assert "documents.id =" in sql
    assert "documents.document_type =" in sql
    assert "documents.fiscal_year =" in sql
    assert "LIMIT" in sql
    assert 7 in compiled.params.values()


@pytest.mark.parametrize(
    ("query", "company", "top_k", "message"),
    [
        ("", "Apple", 5, "query must not be empty"),
        ("revenue", "", 5, "company must not be empty"),
        ("revenue", "Apple", 0, "top_k must be at least 1"),
    ],
)
def test_search_validates_inputs(
    query: str, company: str, top_k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        VectorStore(FakeEmbeddingClient()).search(
            FakeSession([]),  # type: ignore[arg-type]
            query,
            company=company,
            top_k=top_k,
        )
