"""Shared M8 temporal comparison value objects and provenance."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.retrieval.vector_store import VectorSearchResult
from app.temporal.resolution import TemporalDocumentPair
from app.verification.verifier import EntailmentProbabilities


class SourcePeriod(StrEnum):
    OLDER = "older"
    NEWER = "newer"


class TemporalChangeCategory(StrEnum):
    INCREASED = "increased"
    DECLINED = "declined"
    UNCHANGED = "unchanged"
    CHANGED = "changed"
    NEWLY_ADDED = "newly_added"
    REMOVED = "removed"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TemporalEvidence:
    """One numbered source with explicit period and filing provenance."""

    evidence_id: int
    source_period: SourcePeriod
    result: VectorSearchResult

    def __post_init__(self) -> None:
        if self.evidence_id < 1:
            raise ValueError("evidence ID must be at least 1")


@dataclass(frozen=True)
class TemporalFinancialValue:
    metric: str
    amount: Decimal
    unit: str
    fiscal_year: int
    evidence: TemporalEvidence


@dataclass(frozen=True)
class TemporalNumericChange:
    metric: str
    older: TemporalFinancialValue
    newer: TemporalFinancialValue
    absolute_change: Decimal
    percentage_change: Decimal | None
    category: TemporalChangeCategory
    rounding_places: int

    def as_prompt_block(self) -> str:
        percentage = (
            f"{self.percentage_change}%"
            if self.percentage_change is not None
            else "undefined because the older value is zero"
        )
        return (
            "Deterministic temporal calculation (authoritative; do not recompute):\n"
            f"Metric: {self.metric}\n"
            f"Older value ({self.older.fiscal_year}): {self.older.amount:,} "
            f"{self.older.unit} [{self.older.evidence.evidence_id}]\n"
            f"Newer value ({self.newer.fiscal_year}): {self.newer.amount:,} "
            f"{self.newer.unit} [{self.newer.evidence.evidence_id}]\n"
            f"Absolute change: {self.absolute_change:,} {self.newer.unit}\n"
            f"Percentage change: {percentage}\n"
            f"Classification: {self.category.value}\n"
            f"Rounding: {self.rounding_places} decimal place(s), ROUND_HALF_UP"
        )


@dataclass(frozen=True)
class DisclosureTopic:
    """A deliberately supported disclosure identity and its anchor vocabulary."""

    key: str
    query: str
    anchor_groups: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.query.strip() or not self.anchor_groups:
            raise ValueError("disclosure topic must define a key, query, and anchors")
        if any(not group for group in self.anchor_groups):
            raise ValueError("disclosure anchor groups must not be empty")


@dataclass(frozen=True)
class DisclosurePeriodEvidence:
    document_year: int
    source_period: SourcePeriod
    evidence: tuple[TemporalEvidence, ...]
    complete_anchor_scan: bool


@dataclass(frozen=True)
class DisclosureComparison:
    topic: DisclosureTopic
    documents: TemporalDocumentPair
    older: DisclosurePeriodEvidence
    newer: DisclosurePeriodEvidence
    category: TemporalChangeCategory
    older_entails_newer: EntailmentProbabilities | None = None
    newer_entails_older: EntailmentProbabilities | None = None
    error: str | None = None
