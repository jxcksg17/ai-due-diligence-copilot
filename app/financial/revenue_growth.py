"""Deterministic revenue-growth extraction and calculation."""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Final

from app.retrieval.vector_store import VectorSearchResult


class FinancialInputError(ValueError):
    """Base error for missing, ambiguous, or invalid financial inputs."""


class MissingFinancialInputError(FinancialInputError):
    """Raised when source evidence does not contain required values."""


class AmbiguousFinancialInputError(FinancialInputError):
    """Raised when source evidence contains conflicting values."""


class InvalidFinancialInputError(FinancialInputError):
    """Raised when values cannot produce a meaningful revenue-growth result."""


@dataclass(frozen=True)
class SourceFinancialValue:
    """One filing value and the exact evidence from which it was extracted."""

    metric: str
    fiscal_year: int
    amount: Decimal
    unit: str
    evidence_id: int
    chunk_id: int
    page_number: int
    document_id: int

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise InvalidFinancialInputError("metric must not be empty")
        if not self.unit.strip():
            raise InvalidFinancialInputError("unit must not be empty")
        if not self.amount.is_finite():
            raise InvalidFinancialInputError("amount must be finite")
        if self.amount < 0:
            raise InvalidFinancialInputError("revenue amount must not be negative")
        if self.evidence_id < 1:
            raise InvalidFinancialInputError("evidence_id must be at least 1")


@dataclass(frozen=True)
class RevenueGrowthResult:
    """A deterministic percentage change retaining both source values."""

    current: SourceFinancialValue
    prior: SourceFinancialValue
    percentage_change: Decimal
    rounding_places: int
    formula: str = "((current - prior) / prior) * 100"

    def as_prompt_block(self) -> str:
        """Render an authoritative tool result for grounded interpretation."""
        sign = "+" if self.percentage_change > 0 else ""
        return (
            "Deterministic calculation (authoritative; do not recompute):\n"
            f"Metric: Total net sales growth from {self.prior.fiscal_year} "
            f"to {self.current.fiscal_year}\n"
            f"Current source value: {self.current.amount:,} {self.current.unit} "
            f"[{self.current.evidence_id}]\n"
            f"Prior source value: {self.prior.amount:,} {self.prior.unit} "
            f"[{self.prior.evidence_id}]\n"
            f"Formula: {self.formula}\n"
            f"Calculated result: {sign}{self.percentage_change}%\n"
            f"Rounding: {self.rounding_places} decimal place(s), ROUND_HALF_UP"
        )


def calculate_revenue_growth(
    current: SourceFinancialValue | None,
    prior: SourceFinancialValue | None,
    *,
    rounding_places: int = 1,
) -> RevenueGrowthResult:
    """Calculate year-over-year revenue growth with decimal arithmetic."""
    if current is None:
        raise MissingFinancialInputError("current revenue is required")
    if prior is None:
        raise MissingFinancialInputError("prior revenue is required")
    if current.metric != "total_net_sales" or prior.metric != "total_net_sales":
        raise InvalidFinancialInputError("both inputs must be total_net_sales")
    if current.unit != prior.unit:
        raise InvalidFinancialInputError("revenue inputs must use the same unit")
    if current.fiscal_year <= prior.fiscal_year:
        raise InvalidFinancialInputError("current fiscal year must follow prior fiscal year")
    if prior.amount <= 0:
        raise InvalidFinancialInputError("prior revenue must be greater than zero")
    if rounding_places < 0:
        raise InvalidFinancialInputError("rounding_places must not be negative")

    quantum = Decimal("1").scaleb(-rounding_places)
    percentage_change = (
        ((current.amount - prior.amount) / prior.amount) * Decimal("100")
    ).quantize(quantum, rounding=ROUND_HALF_UP)
    return RevenueGrowthResult(
        current=current,
        prior=prior,
        percentage_change=percentage_change,
        rounding_places=rounding_places,
    )


_WITH_CHANGE_ROW: Final = re.compile(
    r"Total\s+net\s+sales\s+\$?\s*(?P<current>\d[\d,]*)\s+"
    r"(?:\(?\d+(?:\.\d+)?\)?|—)\s*%\s*\$?\s*(?P<prior>\d[\d,]*)",
    re.IGNORECASE,
)
_SIMPLE_ROW: Final = re.compile(
    r"Total\s+net\s+sales\s+\$?\s*(?P<current>\d[\d,]*)\s+"
    r"\$?\s*(?P<prior>\d[\d,]*)",
    re.IGNORECASE,
)
_MILLIONS_UNIT: Final = re.compile(r"(?:dollars\s+)?in\s+millions", re.IGNORECASE)


def extract_revenue_growth_inputs(
    evidence: list[VectorSearchResult],
    *,
    current_year: int,
    prior_year: int,
) -> tuple[SourceFinancialValue, SourceFinancialValue]:
    """Extract an explicit total-net-sales pair from ranked filing evidence.

    This deliberately recognizes only the filing table shapes currently needed
    for the MVP. It does not attempt general financial-statement extraction.
    """
    candidates: list[tuple[Decimal, Decimal, int, VectorSearchResult]] = []
    for evidence_id, result in enumerate(evidence, start=1):
        text = result.text
        current_position = text.find(str(current_year))
        prior_position = text.find(str(prior_year))
        if (
            current_position < 0
            or prior_position < 0
            or current_position >= prior_position
            or not _MILLIONS_UNIT.search(text)
        ):
            continue

        match = _WITH_CHANGE_ROW.search(text) or _SIMPLE_ROW.search(text)
        if match is None:
            continue
        try:
            current_amount = Decimal(match.group("current").replace(",", ""))
            prior_amount = Decimal(match.group("prior").replace(",", ""))
        except InvalidOperation as exc:
            raise InvalidFinancialInputError(
                "total net sales row contained an invalid number"
            ) from exc
        candidates.append((current_amount, prior_amount, evidence_id, result))

    if not candidates:
        raise MissingFinancialInputError(
            f"no explicit total net sales values found for {current_year} and {prior_year}"
        )

    distinct_pairs = {(current, prior) for current, prior, _, _ in candidates}
    if len(distinct_pairs) != 1:
        raise AmbiguousFinancialInputError(
            "retrieved evidence contains conflicting total net sales values"
        )

    current_amount, prior_amount, evidence_id, source = candidates[0]
    common = {
        "metric": "total_net_sales",
        "unit": "USD millions",
        "evidence_id": evidence_id,
        "chunk_id": source.chunk_id,
        "page_number": source.page_number,
        "document_id": source.document_id,
    }
    return (
        SourceFinancialValue(
            fiscal_year=current_year,
            amount=current_amount,
            **common,
        ),
        SourceFinancialValue(
            fiscal_year=prior_year,
            amount=prior_amount,
            **common,
        ),
    )
