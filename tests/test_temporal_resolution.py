"""Deterministic two-period filing resolution tests."""

import pytest

from app.retrieval.metadata import DocumentMetadata
from app.temporal.resolution import (
    AmbiguousTemporalDocumentError,
    InvalidTemporalRangeError,
    TemporalDocumentPair,
    TemporalResolutionError,
    UnavailableTemporalDocumentError,
    resolve_temporal_documents,
)


def _document(document_id: int, year: int, *, company: str = "Apple", document_type: str = "10-K") -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        company=company,
        document_type=document_type,
        fiscal_year=year,
    )


def test_resolves_exact_two_periods_for_same_company_and_type() -> None:
    pair = resolve_temporal_documents(
        [_document(2, 2024), _document(1, 2025)],
        company="apple",
        document_type="10-k",
        older_year=2024,
        newer_year=2025,
    )
    assert pair.older.document_id == 2
    assert pair.newer.document_id == 1
    assert pair.company == "Apple"


def test_pair_rejects_different_company_or_document_type() -> None:
    with pytest.raises(TemporalResolutionError, match="one company"):
        TemporalDocumentPair(
            older=_document(2, 2024),
            newer=_document(1, 2025, company="Microsoft"),
        )
    with pytest.raises(TemporalResolutionError, match="same document type"):
        TemporalDocumentPair(
            older=_document(2, 2024),
            newer=_document(1, 2025, document_type="10-Q"),
        )


def test_unavailable_year_is_explicit() -> None:
    with pytest.raises(UnavailableTemporalDocumentError, match="2023"):
        resolve_temporal_documents(
            [_document(2, 2024), _document(1, 2025)],
            company="Apple",
            document_type="10-K",
            older_year=2023,
            newer_year=2025,
        )


def test_duplicate_catalog_period_is_ambiguous() -> None:
    with pytest.raises(AmbiguousTemporalDocumentError, match="2024"):
        resolve_temporal_documents(
            [_document(2, 2024), _document(3, 2024), _document(1, 2025)],
            company="Apple",
            document_type="10-K",
            older_year=2024,
            newer_year=2025,
        )


def test_invalid_period_order_is_rejected() -> None:
    with pytest.raises(InvalidTemporalRangeError, match="precede"):
        resolve_temporal_documents(
            [_document(2, 2024), _document(1, 2025)],
            company="Apple",
            document_type="10-K",
            older_year=2025,
            newer_year=2024,
        )
