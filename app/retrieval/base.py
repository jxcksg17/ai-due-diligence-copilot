"""Shared retrieval interface used by generation services."""

from typing import Protocol

from sqlalchemy.orm import Session

from app.retrieval.vector_store import VectorSearchResult


class Retriever(Protocol):
    """Retrieve semantically relevant chunks within an optional scope."""

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
        """Return ranked chunks with their document metadata."""
