"""Persistence orchestration for filling missing chunk embeddings."""

from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.embeddings.client import EmbeddingClient


def backfill_missing_embeddings(
    db: Session, client: EmbeddingClient, *, batch_size: int = 100
) -> int:
    """Embed null-only chunks, committing each bounded batch.

    Committed batches remain durable if a later provider call fails; the
    uncommitted batch is rolled back and can be safely retried because the
    selection is always ``embedding IS NULL``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    written = 0
    while True:
        chunks = (
            db.query(Chunk)
            .filter(Chunk.embedding.is_(None))
            .order_by(Chunk.id)
            .limit(batch_size)
            .all()
        )
        if not chunks:
            return written

        try:
            for chunk in chunks:
                chunk.embedding = client.embed(chunk.text)
            db.commit()
        except Exception:
            db.rollback()
            raise

        written += len(chunks)
