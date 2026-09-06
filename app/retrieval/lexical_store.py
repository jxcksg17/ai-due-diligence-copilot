"""PostgreSQL-native full-text retrieval over document chunks."""

import re

from sqlalchemy import Select, func, literal_column, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Company, Document
from app.retrieval.vector_store import VectorSearchResult


_ENGLISH = literal_column("'english'::regconfig")


class LexicalStore:
    """Retrieve chunks with PostgreSQL full-text ranking."""

    def search(
        self,
        db: Session,
        query: str,
        *,
        company: str | None,
        top_k: int = 20,
        document_id: int | None = None,
        document_type: str | None = None,
        fiscal_year: int | None = None,
    ) -> list[VectorSearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if company is None or not company.strip():
            raise ValueError("company must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        lexical_query = _build_websearch_query(query, company=company)
        statement = self._build_statement(
            lexical_query,
            company=company,
            top_k=top_k,
            document_id=document_id,
            document_type=document_type,
            fiscal_year=fiscal_year,
        )
        rows = db.execute(statement).all()
        return [
            VectorSearchResult(
                chunk_id=row.chunk_id,
                text=row.text,
                page_number=row.page_number,
                document_id=row.document_id,
                company=row.company,
                document_type=row.document_type,
                fiscal_year=row.fiscal_year,
                cosine_distance=None,
                similarity=None,
                lexical_rank=rank,
                lexical_score=float(row.lexical_score),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    @staticmethod
    def _build_statement(
        query: str,
        *,
        company: str,
        top_k: int,
        document_id: int | None,
        document_type: str | None,
        fiscal_year: int | None,
    ) -> Select[tuple]:
        document_vector = func.to_tsvector(_ENGLISH, Chunk.text)
        search_query = func.websearch_to_tsquery(_ENGLISH, query)
        lexical_score = func.ts_rank_cd(document_vector, search_query, 32).label(
            "lexical_score"
        )
        statement = (
            select(
                Chunk.id.label("chunk_id"),
                Chunk.text,
                Chunk.page_number,
                Document.id.label("document_id"),
                Company.name.label("company"),
                Document.document_type,
                Document.fiscal_year,
                lexical_score,
            )
            .join(Document, Chunk.document_id == Document.id)
            .join(Company, Document.company_id == Company.id)
            .where(
                document_vector.op("@@")(search_query),
                Company.name == company,
            )
            .order_by(lexical_score.desc(), Chunk.id)
            .limit(top_k)
        )
        if document_id is not None:
            statement = statement.where(Document.id == document_id)
        if document_type is not None:
            statement = statement.where(Document.document_type == document_type)
        if fiscal_year is not None:
            statement = statement.where(Document.fiscal_year == fiscal_year)
        return statement


def _build_websearch_query(query: str, *, company: str) -> str:
    """Build a safe disjunctive query for broad lexical candidate recall."""
    company_terms = {term.casefold() for term in re.findall(r"[A-Za-z0-9]+", company)}
    terms: list[str] = []
    seen: set[str] = set()
    for term in re.findall(r"[A-Za-z0-9]+", query):
        normalized = term.casefold()
        if normalized in company_terms or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(term)
    if not terms:
        raise ValueError("query must contain lexical terms beyond company metadata")
    return " OR ".join(terms)
