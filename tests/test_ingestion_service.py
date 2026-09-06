"""
Integration tests for `ingest_document`, covering parsing + chunking +
persistence end to end.

Uses an in-memory SQLite database rather than real PostgreSQL. This is
a deliberate trade-off, not an oversight: the models in `app.db.models`
use plain SQLAlchemy types with no Postgres-specific features yet (no
pgvector, no Postgres-only column types), so SQLite is a faithful
enough stand-in to exercise foreign keys, the unique constraint, and
chunk counts — fast and with no external service required. A
real-Postgres integration test becomes worth adding once local Docker
tooling (a later milestone) gives us a disposable Postgres to run
against; at that point this file's coverage and that one's would be
complementary, not redundant, since Postgres-specific behavior
(pgvector similarity search, for instance) can't be verified on
SQLite at all.
"""

from pathlib import Path

import pytest
from fpdf import FPDF
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import Chunk, Company, Document
from app.ingestion.service import get_or_create_company, ingest_document


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine)

    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Total revenue for fiscal year 2025 was strong.")

    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Operating margin declined slightly versus the prior year.")

    output_path = tmp_path / "acme_2025.pdf"
    pdf.output(str(output_path))
    return output_path


def test_ingest_document_creates_company_document_and_chunks(
    db_session: Session, sample_pdf: Path
) -> None:
    document = ingest_document(
        db_session,
        company_name="Acme Corp",
        document_type="annual_report",
        fiscal_year=2025,
        pdf_path=sample_pdf,
    )

    assert document.id is not None
    assert document.document_type == "annual_report"
    assert document.fiscal_year == 2025

    company = db_session.query(Company).filter_by(name="Acme Corp").one()
    assert document.company_id == company.id

    chunks = db_session.query(Chunk).filter_by(document_id=document.id).all()
    # One short chunk per page, since each page's text is well under
    # CHUNK_SIZE_CHARS in this fixture.
    assert len(chunks) == 2
    assert {c.page_number for c in chunks} == {1, 2}


def test_get_or_create_company_is_idempotent(db_session: Session) -> None:
    first = get_or_create_company(db_session, "Acme Corp")
    db_session.commit()

    second = get_or_create_company(db_session, "Acme Corp")

    assert first.id == second.id
    assert db_session.query(Company).count() == 1


def test_ingesting_duplicate_document_identity_is_rejected(
    db_session: Session, sample_pdf: Path
) -> None:
    """The (company, document_type, fiscal_year) uniqueness constraint
    should be enforced by the database, proving the schema decision in
    app/db/models.py actually does what it's meant to do.
    """
    ingest_document(
        db_session,
        company_name="Acme Corp",
        document_type="annual_report",
        fiscal_year=2025,
        pdf_path=sample_pdf,
    )

    with pytest.raises(IntegrityError):
        ingest_document(
            db_session,
            company_name="Acme Corp",
            document_type="annual_report",
            fiscal_year=2025,
            pdf_path=sample_pdf,
        )
