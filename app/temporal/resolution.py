"""Exact two-period filing resolution without weakening M5 query routing."""

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.retrieval.metadata import DocumentMetadata, MetadataCatalog


class TemporalResolutionError(ValueError):
    """Base error for invalid or unavailable temporal document scopes."""


class InvalidTemporalRangeError(TemporalResolutionError):
    """Raised when the requested period ordering is not meaningful."""


class UnavailableTemporalDocumentError(TemporalResolutionError):
    """Raised when one of the requested filings is not in the catalog."""


class AmbiguousTemporalDocumentError(TemporalResolutionError):
    """Raised when a requested period has more than one matching filing."""


@dataclass(frozen=True)
class TemporalDocumentPair:
    """Two distinct filings for one company and document type."""

    older: DocumentMetadata
    newer: DocumentMetadata

    def __post_init__(self) -> None:
        if self.older.company.casefold() != self.newer.company.casefold():
            raise TemporalResolutionError("temporal documents must belong to one company")
        if self.older.document_type.casefold() != self.newer.document_type.casefold():
            raise TemporalResolutionError(
                "temporal documents must have the same document type"
            )
        if self.older.fiscal_year >= self.newer.fiscal_year:
            raise InvalidTemporalRangeError(
                "older fiscal year must precede newer fiscal year"
            )
        if self.older.document_id == self.newer.document_id:
            raise TemporalResolutionError("temporal comparison requires two documents")

    @property
    def company(self) -> str:
        return self.older.company

    @property
    def document_type(self) -> str:
        return self.older.document_type


def resolve_temporal_documents(
    documents: list[DocumentMetadata],
    *,
    company: str,
    document_type: str,
    older_year: int,
    newer_year: int,
) -> TemporalDocumentPair:
    """Resolve exactly one filing per requested year from the real catalog."""
    if not company.strip():
        raise TemporalResolutionError("company must not be empty")
    if not document_type.strip():
        raise TemporalResolutionError("document type must not be empty")
    if older_year >= newer_year:
        raise InvalidTemporalRangeError(
            "older fiscal year must precede newer fiscal year"
        )

    def resolve_year(year: int) -> DocumentMetadata:
        matches = [
            document
            for document in documents
            if document.company.casefold() == company.casefold()
            and document.document_type.casefold() == document_type.casefold()
            and document.fiscal_year == year
        ]
        if not matches:
            raise UnavailableTemporalDocumentError(
                f"{company} {document_type} for fiscal year {year} is unavailable"
            )
        if len(matches) > 1:
            raise AmbiguousTemporalDocumentError(
                f"{company} {document_type} for fiscal year {year} is ambiguous"
            )
        return matches[0]

    return TemporalDocumentPair(
        older=resolve_year(older_year),
        newer=resolve_year(newer_year),
    )


class TemporalDocumentResolver:
    """Database-backed resolver dedicated to deliberate two-period requests."""

    def __init__(self, catalog: MetadataCatalog | None = None) -> None:
        self._catalog = catalog or MetadataCatalog()

    def resolve(
        self,
        db: Session,
        *,
        company: str,
        document_type: str,
        older_year: int,
        newer_year: int,
    ) -> TemporalDocumentPair:
        return resolve_temporal_documents(
            self._catalog.list_documents(db),
            company=company,
            document_type=document_type,
            older_year=older_year,
            newer_year=newer_year,
        )
