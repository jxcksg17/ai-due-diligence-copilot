"""Deterministic extraction of attributable claims from official issuer releases."""

import re
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.claim_evidence.models import (
    ClaimSource,
    ManagementClaim,
    ManagementClaimType,
)
from app.db.models import Chunk, Company, Document
from app.retrieval.vector_store import VectorSearchResult


class ClaimSourceError(ValueError):
    """Base error for missing, ambiguous, or invalid management sources."""


class MissingClaimSourceError(ClaimSourceError):
    pass


class AmbiguousClaimSourceError(ClaimSourceError):
    pass


class InvalidClaimSourceError(ClaimSourceError):
    pass


@dataclass(frozen=True)
class EarningsReleaseSource:
    document_id: int
    company: str
    document_type: str
    fiscal_year: int
    source_path: str
    chunks: tuple[VectorSearchResult, ...]


_CFO_QUOTE = re.compile(
    r"[“\"](?P<quote>Our September quarter results capped off.*?"
    r"double-digit EPS\s+growth),?[”\"]\s+said\s+Kevan Parekh,\s+"
    r"Apple['’]s CFO\.",
    re.IGNORECASE | re.DOTALL,
)
_CFO_FOLLOWUP = re.compile(
    r"[“\"](?P<quote>And thanks to.*?geographic segments\.)[”\"]",
    re.IGNORECASE | re.DOTALL,
)
_REVENUE_AMOUNT = re.compile(r"revenue reaching \$(?P<amount>\d+(?:\.\d+)?) billion", re.I)
_OFFICIAL_EXHIBIT_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/"
    "000032019325000077/a8-kex991q4202509272025.htm"
)


def _normalize_pdf_text(text: str) -> str:
    normalized = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    repairs = {
        "Apple’ s": "Apple’s",
        "Apple' s": "Apple's",
        "of f": "off",
        "record fiscal year ,": "record fiscal year,",
        "loyalty ,": "loyalty,",
        "W atch": "Watch",
        "T INO": "TINO",
    }
    for broken, repaired in repairs.items():
        normalized = normalized.replace(broken, repaired)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:])", r"\1", normalized)
    return normalized.strip()


def _source_from_match(
    source: EarningsReleaseSource,
    *,
    result: VectorSearchResult,
    quote: str,
) -> ClaimSource:
    return ClaimSource(
        document_id=source.document_id,
        company=source.company,
        document_type=source.document_type,
        fiscal_year=source.fiscal_year,
        source_path=source.source_path,
        source_url=_OFFICIAL_EXHIBIT_URL,
        source_title="Apple reports fourth quarter results",
        source_date="2025-10-30",
        exhibit_label="SEC Exhibit 99.1",
        page_number=result.page_number,
        chunk_ids=(result.chunk_id,),
        speaker_name="Kevan Parekh",
        speaker_role="Chief Financial Officer",
        quote_text=quote,
        evidence=result,
    )


def extract_supported_claims(source: EarningsReleaseSource) -> tuple[ManagementClaim, ...]:
    """Extract only the approved, explicitly attributed CFO claim patterns."""
    if not source.chunks:
        raise MissingClaimSourceError("earnings release contains no chunks")
    if source.document_type != "earnings_release":
        raise InvalidClaimSourceError("source document is not an earnings release")

    claims: list[ManagementClaim] = []
    cfo_result: VectorSearchResult | None = None
    cfo_quote: str | None = None
    followup_quote: str | None = None
    for result in source.chunks:
        text = _normalize_pdf_text(result.text)
        match = _CFO_QUOTE.search(text)
        if match is not None:
            cfo_result = result
            cfo_quote = match.group("quote")
            followup = _CFO_FOLLOWUP.search(text[match.end() :])
            followup_quote = followup.group("quote") if followup is not None else None
            break

    if cfo_result is None or cfo_quote is None:
        raise InvalidClaimSourceError(
            "the expected attributed Apple CFO statement was not found"
        )
    normalized_source = " ".join(_normalize_pdf_text(item.text) for item in source.chunks)
    if "Exhibit 99.1" not in normalized_source:
        raise InvalidClaimSourceError("source is not identifiable as Exhibit 99.1")
    if _OFFICIAL_EXHIBIT_URL not in normalized_source:
        raise InvalidClaimSourceError("source does not retain the official SEC exhibit URL")

    provenance = _source_from_match(source, result=cfo_result, quote=cfo_quote)
    amount_match = _REVENUE_AMOUNT.search(cfo_quote)
    if amount_match is None:
        raise InvalidClaimSourceError("CFO quote did not contain its stated revenue")
    amount_billions = Decimal(amount_match.group("amount"))
    amount_millions = amount_billions * Decimal("1000")
    precision_millions = (
        Decimal("1000")
        if amount_billions == amount_billions.to_integral()
        else Decimal("100")
    )

    exact_revenue_clause = amount_match.group(0)
    claims.append(
        ManagementClaim(
            claim_id="fy2025_revenue_amount",
            original_text=exact_revenue_clause,
            normalized_text=f"Fiscal 2025 revenue reached ${amount_billions} billion.",
            claim_type=ManagementClaimType.REVENUE_AMOUNT,
            source=provenance,
            metric="total_net_sales",
            stated_amount_millions=amount_millions,
            display_precision_millions=precision_millions,
        )
    )
    claims.append(
        ManagementClaim(
            claim_id="record_year_compound_performance",
            original_text=cfo_quote,
            normalized_text=(
                f"Fiscal 2025 was a record fiscal year, with revenue reaching "
                f"${amount_billions} billion and double-digit EPS growth."
            ),
            claim_type=ManagementClaimType.COMPOUND_PERFORMANCE,
            source=provenance,
            metric="total_net_sales",
            stated_amount_millions=amount_millions,
            display_precision_millions=precision_millions,
        )
    )
    claims.append(
        ManagementClaim(
            claim_id="double_digit_eps_growth",
            original_text="double-digit EPS growth",
            normalized_text="Fiscal 2025 EPS growth was double-digit.",
            claim_type=ManagementClaimType.AMBIGUOUS_FINANCIAL,
            source=provenance,
        )
    )

    if followup_quote is not None:
        followup_source = _source_from_match(
            source, result=cfo_result, quote=followup_quote
        )
        claims.extend(
            (
                ManagementClaim(
                    claim_id="customer_satisfaction_and_loyalty",
                    original_text=(
                        "our very high levels of customer satisfaction and loyalty"
                    ),
                    normalized_text=(
                        "Apple reported very high customer satisfaction and loyalty."
                    ),
                    claim_type=ManagementClaimType.QUALITATIVE,
                    source=followup_source,
                ),
                ManagementClaim(
                    claim_id="installed_base_record",
                    original_text=(
                        "our installed base of active devices also reached a new "
                        "all-time high across all product categories and geographic segments"
                    ),
                    normalized_text=(
                        "Apple's installed base reached an all-time high across all "
                        "product categories and geographic segments."
                    ),
                    claim_type=ManagementClaimType.OBJECTIVE_UNAVAILABLE,
                    source=followup_source,
                ),
            )
        )
    return tuple(claims)


class ManagementClaimExtractor:
    """Resolve one official earnings release and extract constrained claims."""

    def load_source(
        self, db: Session, *, company: str, fiscal_year: int
    ) -> EarningsReleaseSource:
        documents = db.execute(
            select(Document, Company.name)
            .join(Company, Document.company_id == Company.id)
            .where(
                func.lower(Company.name) == company.casefold(),
                Document.document_type == "earnings_release",
                Document.fiscal_year == fiscal_year,
            )
        ).all()
        if not documents:
            raise MissingClaimSourceError(
                f"{company} FY{fiscal_year} earnings release is unavailable"
            )
        if len(documents) > 1:
            raise AmbiguousClaimSourceError(
                f"{company} FY{fiscal_year} earnings release is ambiguous"
            )
        document, company_name = documents[0]
        rows = db.execute(
            select(Chunk)
            .where(Chunk.document_id == document.id)
            .order_by(Chunk.chunk_index)
        ).scalars().all()
        results = tuple(
            VectorSearchResult(
                chunk_id=chunk.id,
                text=chunk.text,
                page_number=chunk.page_number,
                document_id=document.id,
                company=company_name,
                document_type=document.document_type,
                fiscal_year=document.fiscal_year,
                cosine_distance=None,
                similarity=None,
            )
            for chunk in rows
        )
        return EarningsReleaseSource(
            document_id=document.id,
            company=company_name,
            document_type=document.document_type,
            fiscal_year=document.fiscal_year,
            source_path=document.source_path,
            chunks=results,
        )

    def extract(
        self, db: Session, *, company: str, fiscal_year: int
    ) -> tuple[ManagementClaim, ...]:
        return extract_supported_claims(
            self.load_source(db, company=company, fiscal_year=fiscal_year)
        )
