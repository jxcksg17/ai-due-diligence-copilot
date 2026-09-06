"""Deterministic topic-presence and semantic-alignment tests."""

from types import SimpleNamespace

from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.temporal.disclosures import (
    DisclosureEvidenceLocator,
    SUPPLY_CHAIN_RISK,
    classify_disclosure_change,
    text_matches_topic,
)
from app.temporal.models import (
    DisclosurePeriodEvidence,
    SourcePeriod,
    TemporalChangeCategory,
    TemporalEvidence,
)
from app.verification.verifier import EntailmentProbabilities


ENTAILS = EntailmentProbabilities(contradiction=0.01, entailment=0.95, neutral=0.04)
NEUTRAL = EntailmentProbabilities(contradiction=0.02, entailment=0.08, neutral=0.90)
CONTRADICTS = EntailmentProbabilities(contradiction=0.90, entailment=0.02, neutral=0.08)


def _result(chunk_id: int, year: int, text: str, *, document_id: int | None = None) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        page_number=10,
        document_id=document_id or chunk_id,
        company="Apple",
        document_type="10-K",
        fiscal_year=year,
        cosine_distance=None,
        similarity=None,
    )


def _period(
    period: SourcePeriod,
    year: int,
    texts: list[str],
    *,
    complete: bool = True,
) -> DisclosurePeriodEvidence:
    offset = 1 if period == SourcePeriod.OLDER else 10
    return DisclosurePeriodEvidence(
        document_year=year,
        source_period=period,
        evidence=tuple(
            TemporalEvidence(
                evidence_id=offset + index,
                source_period=period,
                result=_result(offset + index, year, text),
            )
            for index, text in enumerate(texts)
        ),
        complete_anchor_scan=complete,
    )


class FakeVerifier:
    model_name = "fake-nli"

    def __init__(self, predictions=None, error: Exception | None = None) -> None:
        self.predictions = predictions or []
        self.error = error
        self.pairs = []

    def verify_many(self, pairs):
        self.pairs = pairs
        if self.error:
            raise self.error
        return self.predictions


def test_topic_requires_every_anchor_group() -> None:
    assert text_matches_topic(
        "Suppliers manufacture components that may face shortages.",
        SUPPLY_CHAIN_RISK,
    )
    assert not text_matches_topic("Suppliers provide services.", SUPPLY_CHAIN_RISK)


def test_newly_appearing_and_removed_disclosures_require_complete_scan() -> None:
    empty_old = _period(SourcePeriod.OLDER, 2024, [])
    present_new = _period(
        SourcePeriod.NEWER,
        2025,
        ["Suppliers manufacture components subject to shortages."],
    )
    category, _, _, _ = classify_disclosure_change(
        older=empty_old,
        newer=present_new,
        verifier=FakeVerifier(),
    )
    assert category == TemporalChangeCategory.NEWLY_ADDED

    category, _, _, _ = classify_disclosure_change(
        older=present_new,
        newer=empty_old,
        verifier=FakeVerifier(),
    )
    assert category == TemporalChangeCategory.REMOVED

    incomplete = _period(SourcePeriod.OLDER, 2024, [], complete=False)
    category, _, _, _ = classify_disclosure_change(
        older=incomplete,
        newer=present_new,
        verifier=FakeVerifier(),
    )
    assert category == TemporalChangeCategory.UNAVAILABLE


def test_mutual_entailment_means_semantically_unchanged() -> None:
    older = _period(SourcePeriod.OLDER, 2024, ["Components may face shortages."])
    newer = _period(SourcePeriod.NEWER, 2025, ["Component shortages may occur."])
    verifier = FakeVerifier([ENTAILS, ENTAILS])
    category, forward, reverse, error = classify_disclosure_change(
        older=older,
        newer=newer,
        verifier=verifier,
    )
    assert category == TemporalChangeCategory.UNCHANGED
    assert forward == ENTAILS and reverse == ENTAILS
    assert error is None
    assert len(verifier.pairs) == 2


def test_directional_entailment_or_contradiction_means_changed() -> None:
    older = _period(SourcePeriod.OLDER, 2024, ["Components may face shortages."])
    newer = _period(
        SourcePeriod.NEWER,
        2025,
        ["Components may face shortages and new export restrictions."],
    )
    category, _, _, _ = classify_disclosure_change(
        older=older,
        newer=newer,
        verifier=FakeVerifier([NEUTRAL, ENTAILS]),
    )
    assert category == TemporalChangeCategory.CHANGED

    category, _, _, _ = classify_disclosure_change(
        older=older,
        newer=newer,
        verifier=FakeVerifier([CONTRADICTS, NEUTRAL]),
    )
    assert category == TemporalChangeCategory.CHANGED


def test_two_neutral_directions_are_ambiguous() -> None:
    category, _, _, error = classify_disclosure_change(
        older=_period(SourcePeriod.OLDER, 2024, ["Old disclosure"]),
        newer=_period(SourcePeriod.NEWER, 2025, ["Different disclosure"]),
        verifier=FakeVerifier([NEUTRAL, NEUTRAL]),
    )
    assert category == TemporalChangeCategory.AMBIGUOUS
    assert error is None


def test_verifier_failure_falls_back_to_ambiguous() -> None:
    category, forward, reverse, error = classify_disclosure_change(
        older=_period(SourcePeriod.OLDER, 2024, ["Old disclosure"]),
        newer=_period(SourcePeriod.NEWER, 2025, ["New disclosure"]),
        verifier=FakeVerifier(error=RuntimeError("model unavailable")),
    )
    assert category == TemporalChangeCategory.AMBIGUOUS
    assert forward is None and reverse is None
    assert error == "model unavailable"


class FakeScalars:
    def __init__(self, chunks) -> None:
        self._chunks = chunks

    def scalars(self):
        return self

    def all(self):
        return self._chunks


class FakeDB:
    def __init__(self, chunks) -> None:
        self.chunks = chunks

    def execute(self, _statement):
        return FakeScalars(self.chunks)


class FakeRetriever:
    def __init__(self, results) -> None:
        self.results = results

    def search(self, *_args, **_kwargs):
        return self.results


def test_locator_deduplicates_identical_disclosures_and_preserves_scope() -> None:
    text = "Suppliers manufacture components subject to shortages."
    chunks = [
        SimpleNamespace(id=1, text=text, page_number=10, chunk_index=1),
        SimpleNamespace(id=2, text=text, page_number=11, chunk_index=2),
    ]
    ranked = [_result(1, 2024, text, document_id=7), _result(2, 2024, text, document_id=7)]
    located = DisclosureEvidenceLocator(FakeRetriever(ranked)).locate(
        FakeDB(chunks),
        document=DocumentMetadata(7, "Apple", "10-K", 2024),
        source_period=SourcePeriod.OLDER,
        topic=SUPPLY_CHAIN_RISK,
        top_k=2,
    )
    assert [result.chunk_id for result in located.results] == [1]
    assert located.complete_anchor_scan is True
    assert located.document.document_id == 7
