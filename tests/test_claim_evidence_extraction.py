"""Deterministic extraction and provenance tests for M10."""

from decimal import Decimal

import pytest

from app.claim_evidence.extraction import (
    EarningsReleaseSource,
    InvalidClaimSourceError,
    MissingClaimSourceError,
    extract_supported_claims,
)
from app.claim_evidence.models import ManagementClaimType
from app.retrieval.vector_store import VectorSearchResult


RELEASE_TEXT = """EX-99.1 Exhibit 99.1 Apple reports fourth quarter results.
https://www.sec.gov/Archives/edgar/data/320193/000032019325000077/a8-kex991q4202509272025.htm
“Our September quarter results capped off a record fiscal year, with revenue
reaching $416 billion, as well as double-digit EPS growth,” said Kevan Parekh,
Apple’s CFO. “And thanks to our very high levels of customer satisfaction and
loyalty, our installed base of active devices also reached a new all-time high
across all product categories and geographic segments.”"""


def _result(text: str = RELEASE_TEXT) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=621,
        text=text,
        page_number=1,
        document_id=3,
        company="Apple",
        document_type="earnings_release",
        fiscal_year=2025,
        cosine_distance=None,
        similarity=None,
    )


def _source(*results: VectorSearchResult) -> EarningsReleaseSource:
    return EarningsReleaseSource(
        document_id=3,
        company="Apple",
        document_type="earnings_release",
        fiscal_year=2025,
        source_path="data/apple_fy2025_q4_earnings_release.pdf",
        chunks=results or (_result(),),
    )


def test_extracts_only_anchored_attributable_claims() -> None:
    claims = extract_supported_claims(_source())
    assert [claim.claim_id for claim in claims] == [
        "fy2025_revenue_amount",
        "record_year_compound_performance",
        "double_digit_eps_growth",
        "customer_satisfaction_and_loyalty",
        "installed_base_record",
    ]
    assert claims[0].stated_amount_millions == Decimal("416000")
    assert claims[0].claim_type == ManagementClaimType.REVENUE_AMOUNT
    assert claims[3].claim_type == ManagementClaimType.QUALITATIVE


def test_claim_source_retains_official_exhibit_provenance() -> None:
    claim = extract_supported_claims(_source())[0]
    source = claim.source
    assert source.document_id == 3
    assert source.document_type == "earnings_release"
    assert source.source_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019325000077/a8-kex991q4202509272025.htm"
    )
    assert source.source_title == "Apple reports fourth quarter results"
    assert source.source_date == "2025-10-30"
    assert source.exhibit_label == "SEC Exhibit 99.1"
    assert source.speaker_name == "Kevan Parekh"
    assert source.speaker_role == "Chief Financial Officer"
    assert source.page_number == 1
    assert source.chunk_ids == (621,)


def test_missing_source_chunks_are_rejected() -> None:
    empty_source = EarningsReleaseSource(
        document_id=3,
        company="Apple",
        document_type="earnings_release",
        fiscal_year=2025,
        source_path="data/apple_fy2025_q4_earnings_release.pdf",
        chunks=(),
    )
    with pytest.raises(MissingClaimSourceError):
        extract_supported_claims(empty_source)


def test_non_exhibit_or_unattributed_text_is_rejected() -> None:
    text = RELEASE_TEXT.replace("Exhibit 99.1", "Press summary")
    with pytest.raises(InvalidClaimSourceError, match="Exhibit 99.1"):
        extract_supported_claims(_source(_result(text)))


def test_missing_attributed_cfo_quote_is_rejected() -> None:
    with pytest.raises(InvalidClaimSourceError, match="CFO statement"):
        extract_supported_claims(_source(_result("EX-99.1 Exhibit 99.1")))
