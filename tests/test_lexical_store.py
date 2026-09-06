"""Tests for PostgreSQL-native lexical retrieval."""

from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from app.retrieval.lexical_store import LexicalStore, _build_websearch_query


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


def test_lexical_search_maps_scores_ranks_and_provenance() -> None:
    db = FakeSession(
        [
            SimpleNamespace(
                chunk_id=141,
                text="Total net sales were $416,161 million.",
                page_number=26,
                document_id=1,
                company="Apple",
                document_type="10-K",
                fiscal_year=2025,
                lexical_score=0.42,
            )
        ]
    )
    results = LexicalStore().search(
        db,  # type: ignore[arg-type]
        '"total net sales"',
        company="Apple",
        document_id=1,
        document_type="10-K",
        fiscal_year=2025,
    )
    assert results[0].chunk_id == 141
    assert results[0].lexical_rank == 1
    assert results[0].lexical_score == pytest.approx(0.42)
    assert results[0].similarity is None
    assert results[0].company == "Apple"


def test_lexical_query_uses_or_recall_and_removes_scoped_company() -> None:
    assert _build_websearch_query(
        "What research and development expenses did Apple report?",
        company="Apple",
    ) == "What OR research OR and OR development OR expenses OR did OR report"


def test_lexical_statement_uses_postgres_fts_and_all_metadata_filters() -> None:
    statement = LexicalStore._build_statement(
        "research and development",
        company="Apple",
        top_k=12,
        document_id=1,
        document_type="10-K",
        fiscal_year=2025,
    )
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "to_tsvector('english'::regconfig, chunks.text)" in sql
    assert "websearch_to_tsquery('english'::regconfig" in sql
    assert "@@" in sql
    assert "ts_rank_cd" in sql
    assert "companies.name =" in sql
    assert "documents.id =" in sql
    assert "documents.document_type =" in sql
    assert "documents.fiscal_year =" in sql
    assert 12 in compiled.params.values()


@pytest.mark.parametrize(
    ("query", "company", "top_k", "message"),
    [
        ("", "Apple", 5, "query must not be empty"),
        ("revenue", None, 5, "company must not be empty"),
        ("revenue", "Apple", 0, "top_k must be at least 1"),
    ],
)
def test_lexical_search_validates_inputs(
    query: str, company: str | None, top_k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LexicalStore().search(
            FakeSession([]),  # type: ignore[arg-type]
            query,
            company=company,
            top_k=top_k,
        )
