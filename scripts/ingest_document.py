"""
CLI entry point for ingesting a single PDF document.

Ingestion is treated as an offline/batch operation (per the
architecture document), so it's a script, not an API endpoint — there
is no online query capability yet for an ingestion API to feed into.

Usage:
    python -m scripts.ingest_document \\
        --company "Acme Corp" \\
        --document-type annual_report \\
        --fiscal-year 2025 \\
        --file path/to/annual_report_2025.pdf
"""

import argparse
from pathlib import Path

from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.ingestion.service import ingest_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF financial document.")
    parser.add_argument("--company", required=True, help="Company name")
    parser.add_argument(
        "--document-type",
        required=True,
        help="e.g. annual_report, 10-K, 10-Q, investor_presentation",
    )
    parser.add_argument("--fiscal-year", required=True, type=int)
    parser.add_argument("--file", required=True, type=Path, help="Path to the PDF file")
    args = parser.parse_args()

    # M2 creates tables directly rather than via Alembic migrations —
    # see the M2 design notes for why that's deferred, not skipped.
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        document = ingest_document(
            db,
            company_name=args.company,
            document_type=args.document_type,
            fiscal_year=args.fiscal_year,
            pdf_path=args.file,
        )
        print(
            f"Ingested document id={document.id} "
            f"({args.company}, {args.document_type}, FY{args.fiscal_year}) — "
            f"{len(document.chunks)} chunks stored."
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
