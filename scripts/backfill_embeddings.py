"""Backfill OpenAI embeddings for chunks that do not yet have one.

Usage:
    python -m scripts.backfill_embeddings
"""

from app.config import get_settings
from app.db.session import SessionLocal
from app.embeddings.backfill import backfill_missing_embeddings
from app.embeddings.client import get_embedding_client


def main() -> None:
    settings = get_settings()
    client = get_embedding_client(settings)

    db = SessionLocal()
    try:
        written = backfill_missing_embeddings(db, client)
        print(f"Backfilled embeddings for {written} chunks.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
