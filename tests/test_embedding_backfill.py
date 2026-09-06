"""Deterministic tests for null-only embedding backfill."""

from collections.abc import Sequence

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import Chunk, Company, Document
from app.embeddings.backfill import backfill_missing_embeddings


class FakeEmbeddingClient:
    """Test-only provider: it never sends a network request."""

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions
        self.inputs: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [0.25] * self.dimensions


@pytest.fixture
def db_session() -> Sequence[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


def _add_document_with_chunks(db: Session) -> tuple[Chunk, Chunk]:
    company = Company(name="Example Co")
    db.add(company)
    db.flush()
    document = Document(
        company_id=company.id,
        document_type="10-K",
        fiscal_year=2025,
        source_path="example.pdf",
    )
    db.add(document)
    db.flush()
    already_embedded = Chunk(
        document_id=document.id,
        chunk_index=0,
        page_number=1,
        text="already embedded",
        embedding=[0.0] * 1024,
    )
    missing_embedding = Chunk(
        document_id=document.id,
        chunk_index=1,
        page_number=1,
        text="needs an embedding",
    )
    db.add_all([already_embedded, missing_embedding])
    db.commit()
    return already_embedded, missing_embedding


def test_backfill_only_writes_chunks_with_null_embeddings(db_session: Session) -> None:
    already_embedded, missing_embedding = _add_document_with_chunks(db_session)
    client = FakeEmbeddingClient()

    written = backfill_missing_embeddings(db_session, client, batch_size=1)

    db_session.refresh(already_embedded)
    db_session.refresh(missing_embedding)
    assert written == 1
    assert client.inputs == ["needs an embedding"]
    assert already_embedded.embedding == [0.0] * 1024
    assert missing_embedding.embedding == [0.25] * 1024


def test_backfill_processes_missing_embeddings_across_batches(
    db_session: Session,
) -> None:
    _, missing_embedding = _add_document_with_chunks(db_session)
    db_session.add_all(
        [
            Chunk(
                document_id=missing_embedding.document_id,
                chunk_index=2,
                page_number=1,
                text="second missing embedding",
            ),
            Chunk(
                document_id=missing_embedding.document_id,
                chunk_index=3,
                page_number=1,
                text="third missing embedding",
            ),
        ]
    )
    db_session.commit()
    client = FakeEmbeddingClient()

    written = backfill_missing_embeddings(db_session, client, batch_size=2)

    assert written == 3
    assert client.inputs == [
        "needs an embedding",
        "second missing embedding",
        "third missing embedding",
    ]
    assert db_session.query(Chunk).filter(Chunk.embedding.is_(None)).count() == 0


def test_backfill_rejects_an_invalid_batch_size(db_session: Session) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        backfill_missing_embeddings(db_session, FakeEmbeddingClient(), batch_size=0)
