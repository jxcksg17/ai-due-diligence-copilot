"""Exact pgvector cosine retrieval over embedded document chunks."""

from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Company, Document
from app.embeddings.client import EmbeddingClient


@dataclass(frozen=True)
class VectorSearchResult:
    """One retrieved chunk with the metadata needed for later citations."""

    chunk_id: int
    text: str
    page_number: int
    document_id: int
    company: str
    document_type: str
    fiscal_year: int
    cosine_distance: float | None
    similarity: float | None
    vector_rank: int | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


class VectorStore:
    """Retrieve nearest chunks using exact pgvector cosine distance."""

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self._embedding_client = embedding_client

    def search(
        self,
        db: Session,
        query: str,
        *,
        company: str | None,
        top_k: int = 5,
        document_id: int | None = None,
        document_type: str | None = None,
        fiscal_year: int | None = None,
    ) -> list[VectorSearchResult]:
        """Return the closest embedded chunks within explicit metadata scope.

        The caller supplies metadata filters directly. This layer does not
        infer filters from natural language or perform hybrid/reranked search.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if company is None or not company.strip():
            raise ValueError("company must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = self._embedding_client.embed_query(query)
        statement = self._build_statement(
            query_embedding,
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
                cosine_distance=float(row.cosine_distance),
                similarity=1.0 - float(row.cosine_distance),
            )
            for row in rows
        ]

    @staticmethod
    def _build_statement(
        query_embedding: list[float],
        *,
        company: str,
        top_k: int,
        document_id: int | None,
        document_type: str | None,
        fiscal_year: int | None,
    ) -> Select[tuple]:
        distance = Chunk.embedding.cosine_distance(query_embedding).label(
            "cosine_distance"
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
                distance,
            )
            .join(Document, Chunk.document_id == Document.id)
            .join(Company, Document.company_id == Company.id)
            .where(Chunk.embedding.is_not(None), Company.name == company)
            .order_by(distance)
            .limit(top_k)
        )

        if document_id is not None:
            statement = statement.where(Document.id == document_id)
        if document_type is not None:
            statement = statement.where(Document.document_type == document_type)
        if fiscal_year is not None:
            statement = statement.where(Document.fiscal_year == fiscal_year)

        return statement
