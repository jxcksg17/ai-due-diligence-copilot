"""Risk Radar snapshot and conservative temporal-state tests."""

import pytest

from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.risk_radar.models import (
    RiskEvidence,
    RiskPresenceState,
    RiskSignal,
    RiskTemporalState,
)
from app.risk_radar.service import (
    RiskRadarDocumentError,
    RiskRadarService,
    align_risk_signals,
)
from app.risk_radar.taxonomy import COMPETITION, SUPPLY_CHAIN_MANUFACTURING
from app.temporal.models import SourcePeriod
from app.verification.verifier import EntailmentProbabilities


ENTAILS = EntailmentProbabilities(0.01, 0.95, 0.04)
NEUTRAL = EntailmentProbabilities(0.02, 0.08, 0.90)
CONTRADICTS = EntailmentProbabilities(0.90, 0.02, 0.08)
DOCUMENTS = [
    DocumentMetadata(2, "Apple", "10-K", 2024),
    DocumentMetadata(1, "Apple", "10-K", 2025),
]


class FakeCatalog:
    def list_documents(self, _db):
        return DOCUMENTS


class FakeVerifier:
    model_name = "fake-nli"

    def __init__(self, predictions=(), error=None):
        self.predictions = list(predictions)
        self.error = error
        self.calls = []

    def verify_many(self, pairs):
        self.calls.append(pairs)
        if self.error:
            raise self.error
        return self.predictions


class FakeDiscovery:
    def __init__(self, disclosed_document_ids):
        self.disclosed_document_ids = set(disclosed_document_ids)
        self.calls = []

    def discover(
        self,
        _db,
        *,
        document,
        source_period,
        topic,
        top_k,
        starting_evidence_id,
    ):
        self.calls.append((document, source_period, topic, top_k, starting_evidence_id))
        if document.document_id not in self.disclosed_document_ids:
            return RiskSignal(
                topic,
                document,
                source_period,
                RiskPresenceState.NOT_FOUND,
                (),
                0,
            )
        result = VectorSearchResult(
            chunk_id=100 + document.document_id,
            text=f"{document.fiscal_year} explicit {topic.label} disclosure.",
            page_number=10,
            document_id=document.document_id,
            company=document.company,
            document_type=document.document_type,
            fiscal_year=document.fiscal_year,
            cosine_distance=None,
            similarity=None,
        )
        evidence = RiskEvidence(
            starting_evidence_id,
            topic.key,
            source_period,
            result,
        )
        return RiskSignal(
            topic,
            document,
            source_period,
            RiskPresenceState.DISCLOSED,
            (evidence,),
            1,
        )


def _service(disclosed, predictions=(), error=None):
    return RiskRadarService(
        discovery=FakeDiscovery(disclosed),
        verifier=FakeVerifier(predictions, error),
        catalog=FakeCatalog(),
    )


def test_snapshot_is_repeatable_and_uses_global_evidence_ids() -> None:
    service = _service({1})
    snapshot = service.snapshot(
        object(),
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        topic_keys=(SUPPLY_CHAIN_MANUFACTURING.key, COMPETITION.key),
        top_k=2,
    )
    assert snapshot.document.document_id == 1
    assert [signal.topic.key for signal in snapshot.signals] == [
        SUPPLY_CHAIN_MANUFACTURING.key,
        COMPETITION.key,
    ]
    assert [signal.evidence[0].evidence_id for signal in snapshot.signals] == [1, 2]


def test_snapshot_rejects_unavailable_document() -> None:
    with pytest.raises(RiskRadarDocumentError, match="2023"):
        _service(set()).snapshot(
            object(),
            company="Apple",
            document_type="10-K",
            fiscal_year=2023,
        )


@pytest.mark.parametrize(
    ("disclosed", "predictions", "expected"),
    [
        ({1}, (), RiskTemporalState.NEWLY_DISCLOSED),
        ({2}, (), RiskTemporalState.NO_LONGER_MATCHED),
        (set(), (), RiskTemporalState.NOT_FOUND_BOTH),
        ({1, 2}, (ENTAILS, ENTAILS), RiskTemporalState.RECURRING_UNCHANGED),
        ({1, 2}, (NEUTRAL, ENTAILS), RiskTemporalState.RECURRING_CHANGED),
        ({1, 2}, (NEUTRAL, NEUTRAL), RiskTemporalState.RECURRING_AMBIGUOUS),
        ({1, 2}, (CONTRADICTS, NEUTRAL), RiskTemporalState.RECURRING_CHANGED),
    ],
)
def test_temporal_presence_and_alignment_states(disclosed, predictions, expected) -> None:
    comparison = _service(disclosed, predictions).compare_topic(
        object(),
        company="Apple",
        document_type="10-K",
        older_year=2024,
        newer_year=2025,
        topic=SUPPLY_CHAIN_MANUFACTURING,
    )
    assert comparison.temporal_state == expected
    assert comparison.documents.older.document_id == 2
    assert comparison.documents.newer.document_id == 1
    if comparison.older.evidence:
        assert comparison.older.evidence[0].source_period == SourcePeriod.OLDER
        assert comparison.older.evidence[0].document_id == 2
    if comparison.newer.evidence:
        assert comparison.newer.evidence[0].source_period == SourcePeriod.NEWER
        assert comparison.newer.evidence[0].document_id == 1


def test_verifier_failure_is_exposed_as_recurring_ambiguous() -> None:
    comparison = _service(
        {1, 2},
        error=RuntimeError("verifier unavailable"),
    ).compare_topic(
        object(),
        company="Apple",
        document_type="10-K",
        older_year=2024,
        newer_year=2025,
        topic=COMPETITION,
    )
    assert comparison.temporal_state == RiskTemporalState.RECURRING_AMBIGUOUS
    assert comparison.error == "verifier unavailable"


def test_discovery_and_nli_alignment_can_run_as_separate_model_stages() -> None:
    service = RiskRadarService(
        discovery=FakeDiscovery({1, 2}),
        catalog=FakeCatalog(),
    )
    signal_pair = service.discover_topic(
        object(),
        company="Apple",
        document_type="10-K",
        older_year=2024,
        newer_year=2025,
        topic=COMPETITION,
    )
    comparison = align_risk_signals(
        signal_pair,
        verifier=FakeVerifier([ENTAILS, ENTAILS]),
    )
    assert comparison.temporal_state == RiskTemporalState.RECURRING_UNCHANGED
