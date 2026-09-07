"""Risk Radar snapshot and conservative two-filing comparison services."""

from sqlalchemy.orm import Session

from app.retrieval.metadata import DocumentMetadata, MetadataCatalog
from app.risk_radar.discovery import RiskEvidenceDiscovery
from app.risk_radar.models import (
    RiskComparison,
    RiskPresenceState,
    RiskRadarSnapshot,
    RiskSignal,
    RiskSignalPair,
    RiskTemporalState,
    TemporalRiskRadar,
)
from app.risk_radar.taxonomy import RiskTopic, resolve_risk_topics
from app.temporal.disclosures import classify_disclosure_change
from app.temporal.models import (
    DisclosurePeriodEvidence,
    SourcePeriod,
    TemporalChangeCategory,
    TemporalEvidence,
)
from app.temporal.resolution import TemporalDocumentResolver
from app.verification.verifier import EntailmentVerifier


class RiskRadarDocumentError(ValueError):
    """Raised when a single requested filing cannot be resolved exactly."""


def _resolve_single_document(
    documents: list[DocumentMetadata],
    *,
    company: str,
    document_type: str,
    fiscal_year: int,
) -> DocumentMetadata:
    matches = [
        document
        for document in documents
        if document.company.casefold() == company.casefold()
        and document.document_type.casefold() == document_type.casefold()
        and document.fiscal_year == fiscal_year
    ]
    if not matches:
        raise RiskRadarDocumentError(
            f"{company} {document_type} for fiscal year {fiscal_year} is unavailable"
        )
    if len(matches) > 1:
        raise RiskRadarDocumentError(
            f"{company} {document_type} for fiscal year {fiscal_year} is ambiguous"
        )
    return matches[0]


def _as_period_evidence(signal: RiskSignal) -> DisclosurePeriodEvidence:
    return DisclosurePeriodEvidence(
        document_year=signal.document.fiscal_year,
        source_period=signal.source_period,
        evidence=tuple(
            TemporalEvidence(
                evidence_id=item.evidence_id,
                source_period=item.source_period,
                result=item.result,
            )
            for item in signal.evidence
        ),
        complete_anchor_scan=signal.complete_item_1a_scan,
    )


def _risk_temporal_state(
    category: TemporalChangeCategory,
    older: RiskSignal,
    newer: RiskSignal,
) -> RiskTemporalState:
    if (
        older.presence == RiskPresenceState.NOT_FOUND
        and newer.presence == RiskPresenceState.NOT_FOUND
    ):
        return RiskTemporalState.NOT_FOUND_BOTH
    mapping = {
        TemporalChangeCategory.NEWLY_ADDED: RiskTemporalState.NEWLY_DISCLOSED,
        TemporalChangeCategory.REMOVED: RiskTemporalState.NO_LONGER_MATCHED,
        TemporalChangeCategory.UNCHANGED: RiskTemporalState.RECURRING_UNCHANGED,
        TemporalChangeCategory.CHANGED: RiskTemporalState.RECURRING_CHANGED,
        TemporalChangeCategory.AMBIGUOUS: RiskTemporalState.RECURRING_AMBIGUOUS,
        TemporalChangeCategory.UNAVAILABLE: RiskTemporalState.INSUFFICIENT,
    }
    return mapping.get(category, RiskTemporalState.INSUFFICIENT)


def align_risk_signals(
    signal_pair: RiskSignalPair,
    *,
    verifier: EntailmentVerifier,
) -> RiskComparison:
    """Align already-discovered signals after retrieval models are released."""
    category, forward, reverse, error = classify_disclosure_change(
        older=_as_period_evidence(signal_pair.older),
        newer=_as_period_evidence(signal_pair.newer),
        verifier=verifier,
    )
    return RiskComparison(
        topic=signal_pair.topic,
        documents=signal_pair.documents,
        older=signal_pair.older,
        newer=signal_pair.newer,
        temporal_state=_risk_temporal_state(
            category,
            signal_pair.older,
            signal_pair.newer,
        ),
        older_entails_newer=forward,
        newer_entails_older=reverse,
        error=error,
    )


class RiskRadarService:
    """Build deterministic filing snapshots and NLI-aligned temporal radars."""

    def __init__(
        self,
        *,
        discovery: RiskEvidenceDiscovery,
        verifier: EntailmentVerifier | None = None,
        catalog: MetadataCatalog | None = None,
        temporal_resolver: TemporalDocumentResolver | None = None,
    ) -> None:
        self._discovery = discovery
        self._verifier = verifier
        self._catalog = catalog or MetadataCatalog()
        self._temporal_resolver = temporal_resolver or TemporalDocumentResolver(
            self._catalog
        )

    def snapshot(
        self,
        db: Session,
        *,
        company: str,
        document_type: str,
        fiscal_year: int,
        topic_keys: tuple[str, ...] | None = None,
        top_k: int = 3,
    ) -> RiskRadarSnapshot:
        document = _resolve_single_document(
            self._catalog.list_documents(db),
            company=company,
            document_type=document_type,
            fiscal_year=fiscal_year,
        )
        signals: list[RiskSignal] = []
        next_evidence_id = 1
        for topic in resolve_risk_topics(topic_keys):
            signal = self._discovery.discover(
                db,
                document=document,
                source_period=SourcePeriod.NEWER,
                topic=topic,
                top_k=top_k,
                starting_evidence_id=next_evidence_id,
            )
            signals.append(signal)
            next_evidence_id += len(signal.evidence)
        return RiskRadarSnapshot(document=document, signals=tuple(signals))

    def compare_topic(
        self,
        db: Session,
        *,
        company: str,
        document_type: str,
        older_year: int,
        newer_year: int,
        topic: RiskTopic,
        top_k: int = 2,
    ) -> RiskComparison:
        if self._verifier is None:
            raise ValueError("temporal Risk Radar requires an entailment verifier")
        signal_pair = self.discover_topic(
            db,
            company=company,
            document_type=document_type,
            older_year=older_year,
            newer_year=newer_year,
            topic=topic,
            top_k=top_k,
        )
        return align_risk_signals(signal_pair, verifier=self._verifier)

    def discover_topic(
        self,
        db: Session,
        *,
        company: str,
        document_type: str,
        older_year: int,
        newer_year: int,
        topic: RiskTopic,
        top_k: int = 2,
    ) -> RiskSignalPair:
        """Discover both periods without loading or invoking an NLI verifier."""
        documents = self._temporal_resolver.resolve(
            db,
            company=company,
            document_type=document_type,
            older_year=older_year,
            newer_year=newer_year,
        )
        older = self._discovery.discover(
            db,
            document=documents.older,
            source_period=SourcePeriod.OLDER,
            topic=topic,
            top_k=top_k,
            starting_evidence_id=1,
        )
        newer = self._discovery.discover(
            db,
            document=documents.newer,
            source_period=SourcePeriod.NEWER,
            topic=topic,
            top_k=top_k,
            starting_evidence_id=len(older.evidence) + 1,
        )
        return RiskSignalPair(
            topic=topic,
            documents=documents,
            older=older,
            newer=newer,
        )

    def temporal_radar(
        self,
        db: Session,
        *,
        company: str,
        document_type: str,
        older_year: int,
        newer_year: int,
        topic_keys: tuple[str, ...] | None = None,
        top_k: int = 2,
    ) -> TemporalRiskRadar:
        topics = resolve_risk_topics(topic_keys)
        comparisons = tuple(
            self.compare_topic(
                db,
                company=company,
                document_type=document_type,
                older_year=older_year,
                newer_year=newer_year,
                topic=topic,
                top_k=top_k,
            )
            for topic in topics
        )
        if not comparisons:
            raise ValueError("temporal Risk Radar requires at least one topic")
        return TemporalRiskRadar(
            documents=comparisons[0].documents,
            comparisons=comparisons,
        )
