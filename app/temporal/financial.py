"""Cross-filing total-net-sales extraction and deterministic comparison."""

import re
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy.orm import Session

from app.financial.revenue_growth import (
    FinancialInputError,
    extract_revenue_growth_inputs,
)
from app.retrieval.base import Retriever
from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.temporal.models import (
    SourcePeriod,
    TemporalChangeCategory,
    TemporalEvidence,
    TemporalFinancialValue,
    TemporalNumericChange,
)
from app.temporal.resolution import TemporalDocumentPair, TemporalDocumentResolver


class TemporalFinancialInputError(ValueError):
    """Raised when two filing-specific values cannot be compared safely."""


def _is_consolidated_operations_statement(text: str) -> bool:
    """Recognize the statement heading despite PDF spacing artifacts."""
    letters_only = re.sub(r"[^a-z]", "", text.casefold())
    return "consolidatedstatementsofoperations" in letters_only


def extract_period_total_net_sales(
    results: list[VectorSearchResult],
    *,
    document: DocumentMetadata,
    source_period: SourcePeriod,
    evidence_id: int,
) -> TemporalFinancialValue:
    """Extract the filing year's company total from its operations statement."""
    statement_results = [
        result
        for result in results
        if _is_consolidated_operations_statement(result.text)
    ]
    if not statement_results:
        raise TemporalFinancialInputError(
            "no Consolidated Statements of Operations evidence was retrieved"
        )
    try:
        current, _ = extract_revenue_growth_inputs(
            statement_results,
            current_year=document.fiscal_year,
            prior_year=document.fiscal_year - 1,
        )
    except FinancialInputError as exc:
        raise TemporalFinancialInputError(str(exc)) from exc

    source = next(
        (
            result
            for result in statement_results
            if result.chunk_id == current.chunk_id
        ),
        None,
    )
    if source is None or source.document_id != document.document_id:
        raise TemporalFinancialInputError(
            "financial evidence does not match the resolved filing"
        )
    temporal_evidence = TemporalEvidence(
        evidence_id=evidence_id,
        source_period=source_period,
        result=source,
    )
    return TemporalFinancialValue(
        metric="total_net_sales",
        amount=current.amount,
        unit=current.unit,
        fiscal_year=document.fiscal_year,
        evidence=temporal_evidence,
    )


def calculate_temporal_numeric_change(
    older: TemporalFinancialValue | None,
    newer: TemporalFinancialValue | None,
    *,
    rounding_places: int = 1,
) -> TemporalNumericChange:
    """Calculate a provenance-preserving cross-filing numeric change."""
    if older is None or newer is None:
        raise TemporalFinancialInputError("both temporal values are required")
    if older.metric != newer.metric:
        raise TemporalFinancialInputError("temporal metrics must match")
    if older.unit != newer.unit:
        raise TemporalFinancialInputError("temporal units must match")
    if older.fiscal_year >= newer.fiscal_year:
        raise TemporalFinancialInputError("older value must precede newer value")
    if older.evidence.result.document_id == newer.evidence.result.document_id:
        raise TemporalFinancialInputError(
            "cross-filing comparison requires distinct documents"
        )
    if rounding_places < 0:
        raise TemporalFinancialInputError("rounding places must not be negative")

    absolute_change = newer.amount - older.amount
    if absolute_change > 0:
        category = TemporalChangeCategory.INCREASED
    elif absolute_change < 0:
        category = TemporalChangeCategory.DECLINED
    else:
        category = TemporalChangeCategory.UNCHANGED

    percentage_change: Decimal | None
    if older.amount == 0:
        percentage_change = None
    else:
        quantum = Decimal("1").scaleb(-rounding_places)
        percentage_change = (
            (absolute_change / older.amount) * Decimal("100")
        ).quantize(quantum, rounding=ROUND_HALF_UP)

    return TemporalNumericChange(
        metric=older.metric,
        older=older,
        newer=newer,
        absolute_change=absolute_change,
        percentage_change=percentage_change,
        category=category,
        rounding_places=rounding_places,
    )


class TemporalFinancialComparisonService:
    """Resolve two filings, retrieve each independently, then calculate."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        resolver: TemporalDocumentResolver | None = None,
    ) -> None:
        self._retriever = retriever
        self._resolver = resolver or TemporalDocumentResolver()

    def compare_total_net_sales(
        self,
        db: Session,
        *,
        company: str,
        older_year: int,
        newer_year: int,
        document_type: str = "10-K",
        top_k: int = 10,
        rounding_places: int = 1,
    ) -> tuple[TemporalDocumentPair, TemporalNumericChange]:
        documents = self._resolver.resolve(
            db,
            company=company,
            document_type=document_type,
            older_year=older_year,
            newer_year=newer_year,
        )
        older_results = self._search_period(db, documents.older, top_k=top_k)
        newer_results = self._search_period(db, documents.newer, top_k=top_k)
        older = extract_period_total_net_sales(
            older_results,
            document=documents.older,
            source_period=SourcePeriod.OLDER,
            evidence_id=1,
        )
        newer = extract_period_total_net_sales(
            newer_results,
            document=documents.newer,
            source_period=SourcePeriod.NEWER,
            evidence_id=2,
        )
        return documents, calculate_temporal_numeric_change(
            older,
            newer,
            rounding_places=rounding_places,
        )

    def _search_period(
        self,
        db: Session,
        document: DocumentMetadata,
        *,
        top_k: int,
    ) -> list[VectorSearchResult]:
        return self._retriever.search(
            db,
            f"total net sales revenue for fiscal year {document.fiscal_year}",
            company=document.company,
            top_k=top_k,
            document_id=document.document_id,
            document_type=document.document_type,
            fiscal_year=document.fiscal_year,
        )
