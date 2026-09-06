"""
Ingestion orchestration.

Kept separate from the CLI script (`scripts/ingest_document.py`) on
purpose: this module contains the actual logic and is unit-testable
without going through argument parsing or process-level concerns.
If a future milestone adds an API endpoint for ingestion, it calls
`ingest_document()` directly rather than duplicating this logic.
"""

from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import Chunk, Company, Document
from app.ingestion.chunking import chunk_document
from app.ingestion.parser import extract_pages


def get_or_create_company(db: Session, name: str) -> Company:
    """Look up a company by name, creating it if it doesn't exist yet.

    `db.flush()` (not `db.commit()`) pushes the INSERT to the database
    and populates `company.id`, without ending the transaction — the
    caller (`ingest_document`) controls when to commit, so a failure
    partway through ingestion rolls back the company creation too
    rather than leaving an orphaned company with no document.
    """
    company = db.query(Company).filter(Company.name == name).one_or_none()
    if company is None:
        company = Company(name=name)
        db.add(company)
        db.flush()
    return company


def ingest_document(
    db: Session,
    *,
    company_name: str,
    document_type: str,
    fiscal_year: int,
    pdf_path: Path,
) -> Document:
    """Parse, chunk, and persist one PDF document.

    Raises sqlalchemy.exc.IntegrityError if a document with the same
    (company, document_type, fiscal_year) already exists — see the
    unique constraint on the Document model. Ingestion is not
    idempotent by design in M2: re-ingesting the same filing is a
    decision left to the caller (e.g. delete-then-reingest), not
    silently handled here.
    """
    company = get_or_create_company(db, company_name)

    document = Document(
        company_id=company.id,
        document_type=document_type,
        fiscal_year=fiscal_year,
        source_path=str(pdf_path),
    )
    db.add(document)
    db.flush()  # populate document.id for the chunks below

    pages = extract_pages(pdf_path)
    text_chunks = chunk_document(pages)

    for text_chunk in text_chunks:
        db.add(
            Chunk(
                document_id=document.id,
                chunk_index=text_chunk.chunk_index,
                page_number=text_chunk.page_number,
                text=text_chunk.text,
            )
        )

    db.commit()
    db.refresh(document)
    return document
