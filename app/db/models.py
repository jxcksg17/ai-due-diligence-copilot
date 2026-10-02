"""
Core ingestion data model: Company -> Document -> Chunk.

This is intentionally the smallest schema that supports M2. It will
grow (financial_facts, claims, eval_cases/eval_runs) in later
milestones, per the architecture document — those tables are not
created here just because the architecture describes them eventually.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    literal_column,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True)
    # Unique because "get or create by name" (used during ingestion) only
    # makes sense if names uniquely identify a company at this stage.
    # A ticker-based identity, or handling company name aliases, is a
    # refinement for later if it turns out to matter.
    name = Column(String(255), nullable=False, unique=True)
    ticker = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    documents = relationship("Document", back_populates="company")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)

    # e.g. "annual_report", "10-K", "10-Q", "investor_presentation",
    # "earnings_transcript". A plain string, not an enum/lookup table,
    # for now — we don't yet know the full set well enough to lock it
    # down, and a typo here is easy to catch by inspection at this scale.
    document_type = Column(String(50), nullable=False)
    fiscal_year = Column(Integer, nullable=False)

    source_path = Column(String(1024), nullable=False)
    ingested_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    company = relationship("Company", back_populates="documents")
    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Enforces "one filing per company, per type, per fiscal year" as
        # a database-level invariant rather than an application-level
        # convention. This is what makes temporal queries later
        # ("compare company X's annual_report across 2024 and 2025")
        # reliable: there is exactly one row to find, guaranteed by the
        # schema, not by hoping ingestion was careful.
        UniqueConstraint(
            "company_id", "document_type", "fiscal_year", name="uq_document_identity"
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)

    # Running index across the whole document (not per-page), so chunks
    # can be reassembled in reading order regardless of page.
    chunk_index = Column(Integer, nullable=False)

    # 1-indexed to match how a human/citation refers to "page 12".
    page_number = Column(Integer, nullable=False)

    text = Column(Text, nullable=False)

    # Populated by the embedding backfill in a later M3 step.  It is
    # intentionally nullable so this additive schema change preserves all
    # existing M2 chunks.
    embedding = Column(Vector(1024), nullable=True)

    # Nullable placeholder for later, smarter section-aware chunking
    # (e.g. "Risk Factors", "MD&A"). Naive fixed-size chunking in M2
    # has no way to know the section, so this is left unset for now
    # rather than guessed at.
    section = Column(String(255), nullable=True)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index(
            "ix_chunks_text_fts",
            func.to_tsvector(literal_column("'english'::regconfig"), text),
            postgresql_using="gin",
        ).ddl_if(dialect="postgresql"),
    )
