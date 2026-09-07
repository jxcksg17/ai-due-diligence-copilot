"""Deterministic Item 1A location and Risk Radar discovery tests."""

from types import SimpleNamespace

import pytest

from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.risk_radar.discovery import RiskEvidenceDiscovery, text_matches_risk_topic
from app.risk_radar.models import RiskPresenceState
from app.risk_radar.section import (
    Item1ASection,
    RiskSectionChunk,
    RiskSectionError,
    extract_item_1a_section,
)
from app.risk_radar.taxonomy import (
    LEGAL_REGULATORY,
    SUPPLY_CHAIN_MANUFACTURING,
    UnsupportedRiskTopicError,
    get_risk_topic,
)
from app.temporal.models import SourcePeriod


DOCUMENT = DocumentMetadata(7, "Apple", "10-K", 2025)


def _chunk(chunk_id: int, index: int, page: int, text: str):
    return SimpleNamespace(
        id=chunk_id,
        chunk_index=index,
        page_number=page,
        text=text,
    )


def _result(chunk_id: int, text: str, *, document_id: int = 7):
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        page_number=10 + chunk_id,
        document_id=document_id,
        company="Apple",
        document_type="10-K",
        fiscal_year=2025,
        cosine_distance=0.1,
        similarity=0.9,
        rerank_score=float(chunk_id),
    )


def test_item_1a_location_excludes_toc_and_clips_boundaries() -> None:
    chunks = [
        _chunk(1, 1, 3, "Table of Contents Item 1A. Risk Factors 5 Item 1B. 17"),
        _chunk(2, 2, 8, "Business tail. Item 1A. Risk Factors Opening risk."),
        _chunk(3, 3, 9, "Middle risk disclosure."),
        _chunk(4, 4, 20, "Final risk disclosure. Item 1B. Unresolved Staff Comments"),
    ]
    section = extract_item_1a_section(chunks, document=DOCUMENT)
    assert [item.chunk_id for item in section.chunks] == [2, 3, 4]
    assert section.chunks[0].text.startswith("Item 1A. Risk Factors")
    assert "Business tail" not in section.chunks[0].text
    assert section.chunks[-1].text == "Final risk disclosure."
    assert section.complete is True


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([_chunk(1, 1, 1, "No risk heading")], "was not found"),
        (
            [
                _chunk(1, 1, 1, "Item 1A. Risk Factors A"),
                _chunk(2, 2, 2, "Item 1A. Risk Factors B"),
                _chunk(3, 3, 3, "Item 1B. End"),
            ],
            "ambiguous",
        ),
        ([_chunk(1, 1, 1, "Item 1A. Risk Factors A")], "end boundary"),
    ],
)
def test_item_1a_location_rejects_incomplete_or_ambiguous_sections(
    chunks, message: str
) -> None:
    with pytest.raises(RiskSectionError, match=message):
        extract_item_1a_section(chunks, document=DOCUMENT)


def test_topic_assignment_is_explicit_and_whole_word_based() -> None:
    assert text_matches_risk_topic(
        "The Company relies on suppliers for manufacturing components.",
        SUPPLY_CHAIN_MANUFACTURING,
    )
    assert not text_matches_risk_topic(
        "The Company discusses supplier relationships.",
        SUPPLY_CHAIN_MANUFACTURING,
    )
    assert text_matches_risk_topic(
        "Legal proceedings and regulatory investigations may increase costs.",
        LEGAL_REGULATORY,
    )
    with pytest.raises(UnsupportedRiskTopicError, match="unsupported risk topic"):
        get_risk_topic("invented_severity_risk")


class FakeSectionLocator:
    def __init__(self, chunks):
        self.chunks = tuple(chunks)

    def locate(self, _db, document):
        return Item1ASection(document=document, chunks=self.chunks)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, *_args, **kwargs):
        self.calls.append(kwargs)
        return self.results


def test_discovery_groups_multiple_passages_deduplicates_and_preserves_scope() -> None:
    first = "Suppliers manufacture components that may face shortages."
    second = "A limited source supplies components used in manufacturing."
    chunks = [
        RiskSectionChunk(1, 1, 10, first),
        RiskSectionChunk(2, 2, 11, first),
        RiskSectionChunk(3, 3, 12, second),
        RiskSectionChunk(4, 4, 13, "Employee retention may be difficult."),
    ]
    retriever = FakeRetriever([_result(3, "unclipped"), _result(1, "unclipped")])
    signal = RiskEvidenceDiscovery(
        retriever,
        section_locator=FakeSectionLocator(chunks),
    ).discover(
        object(),
        document=DOCUMENT,
        source_period=SourcePeriod.NEWER,
        topic=SUPPLY_CHAIN_MANUFACTURING,
        top_k=3,
        starting_evidence_id=5,
    )

    assert signal.presence == RiskPresenceState.DISCLOSED
    assert signal.qualifying_passage_count == 2
    assert [item.chunk_id for item in signal.evidence] == [3, 1]
    assert [item.evidence_id for item in signal.evidence] == [5, 6]
    assert signal.evidence[0].source_text == second
    assert signal.evidence[0].topic_key == SUPPLY_CHAIN_MANUFACTURING.key
    assert signal.evidence[0].document_id == 7
    assert retriever.calls[0]["document_id"] == 7
    assert retriever.calls[0]["fiscal_year"] == 2025


def test_no_matching_disclosure_is_explicit_and_skips_retrieval() -> None:
    retriever = FakeRetriever([])
    signal = RiskEvidenceDiscovery(
        retriever,
        section_locator=FakeSectionLocator(
            [RiskSectionChunk(1, 1, 10, "Employee retention may be difficult.")]
        ),
    ).discover(
        object(),
        document=DOCUMENT,
        source_period=SourcePeriod.NEWER,
        topic=SUPPLY_CHAIN_MANUFACTURING,
    )
    assert signal.presence == RiskPresenceState.NOT_FOUND
    assert signal.evidence == ()
    assert "not proof" in signal.as_signal_text()
    assert retriever.calls == []
