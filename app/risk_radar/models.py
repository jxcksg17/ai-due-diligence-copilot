"""Risk Radar value objects with explicit evidence and signal provenance."""

from dataclasses import dataclass
from enum import StrEnum

from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.risk_radar.taxonomy import RiskTopic
from app.temporal.models import SourcePeriod
from app.temporal.resolution import TemporalDocumentPair
from app.verification.verifier import EntailmentProbabilities


class RiskPresenceState(StrEnum):
    """Observable result of a complete, topic-anchored Item 1A scan."""

    DISCLOSED = "disclosed"
    NOT_FOUND = "not_found"


class RiskTemporalState(StrEnum):
    """Conservative cross-filing state derived from presence and M8 alignment."""

    NEWLY_DISCLOSED = "newly_disclosed"
    NO_LONGER_MATCHED = "no_longer_matched"
    RECURRING_UNCHANGED = "recurring_unchanged"
    RECURRING_CHANGED = "recurring_changed"
    RECURRING_AMBIGUOUS = "recurring_ambiguous"
    NOT_FOUND_BOTH = "not_found_both"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class RiskEvidence:
    """One prompt-visible disclosure with its complete filing provenance."""

    evidence_id: int
    topic_key: str
    source_period: SourcePeriod
    result: VectorSearchResult

    def __post_init__(self) -> None:
        if self.evidence_id < 1:
            raise ValueError("evidence ID must be at least 1")
        if not self.topic_key.strip():
            raise ValueError("risk topic key must not be empty")

    @property
    def company(self) -> str:
        return self.result.company

    @property
    def fiscal_year(self) -> int:
        return self.result.fiscal_year

    @property
    def document_id(self) -> int:
        return self.result.document_id

    @property
    def document_type(self) -> str:
        return self.result.document_type

    @property
    def page_number(self) -> int:
        return self.result.page_number

    @property
    def chunk_id(self) -> int:
        return self.result.chunk_id

    @property
    def source_text(self) -> str:
        return self.result.text


@dataclass(frozen=True)
class RiskSignal:
    """Deterministic filing-level presence signal, separate from interpretation."""

    topic: RiskTopic
    document: DocumentMetadata
    source_period: SourcePeriod
    presence: RiskPresenceState
    evidence: tuple[RiskEvidence, ...]
    qualifying_passage_count: int
    complete_item_1a_scan: bool = True

    def __post_init__(self) -> None:
        if self.qualifying_passage_count < len(self.evidence):
            raise ValueError("qualifying passage count cannot be below evidence count")
        if self.presence == RiskPresenceState.DISCLOSED and not self.evidence:
            raise ValueError("a disclosed signal requires evidence")
        if self.presence == RiskPresenceState.NOT_FOUND and self.evidence:
            raise ValueError("a not-found signal cannot contain evidence")
        for item in self.evidence:
            if item.topic_key != self.topic.key:
                raise ValueError("risk evidence topic does not match its signal")
            if item.source_period != self.source_period:
                raise ValueError("risk evidence period does not match its signal")
            if (
                item.document_id != self.document.document_id
                or item.fiscal_year != self.document.fiscal_year
            ):
                raise ValueError("risk evidence document does not match its signal")

    def as_signal_text(self) -> str:
        if self.presence == RiskPresenceState.DISCLOSED:
            return (
                f"Explicit {self.topic.label} disclosure matched in Item 1A "
                f"({self.qualifying_passage_count} distinct qualifying passage(s))."
            )
        return (
            f"No explicit {self.topic.label} disclosure matched the configured "
            "anchors in the complete Item 1A scan; this is not proof that the "
            "underlying business risk is absent."
        )


@dataclass(frozen=True)
class RiskSignalPair:
    """Evidence-discovery output that can outlive retrieval model resources."""

    topic: RiskTopic
    documents: TemporalDocumentPair
    older: RiskSignal
    newer: RiskSignal


@dataclass(frozen=True)
class RiskComparison:
    """Two filing signals plus conservative semantic temporal alignment."""

    topic: RiskTopic
    documents: TemporalDocumentPair
    older: RiskSignal
    newer: RiskSignal
    temporal_state: RiskTemporalState
    older_entails_newer: EntailmentProbabilities | None = None
    newer_entails_older: EntailmentProbabilities | None = None
    error: str | None = None


@dataclass(frozen=True)
class RiskRadarSnapshot:
    document: DocumentMetadata
    signals: tuple[RiskSignal, ...]


@dataclass(frozen=True)
class TemporalRiskRadar:
    documents: TemporalDocumentPair
    comparisons: tuple[RiskComparison, ...]
