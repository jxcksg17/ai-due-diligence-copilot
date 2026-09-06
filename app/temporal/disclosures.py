"""Topic-anchored disclosure location and semantic temporal alignment."""

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.retrieval.base import Retriever
from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.temporal.models import (
    DisclosureComparison,
    DisclosurePeriodEvidence,
    DisclosureTopic,
    SourcePeriod,
    TemporalChangeCategory,
    TemporalEvidence,
)
from app.temporal.resolution import TemporalDocumentResolver
from app.verification.verifier import EntailmentProbabilities, EntailmentVerifier


SUPPLY_CHAIN_RISK = DisclosureTopic(
    key="supply_chain_risk",
    query="supply chain manufacturing suppliers components shortages",
    anchor_groups=(
        ("supply", "supplier", "suppliers", "source", "sources", "outsourcing"),
        (
            "component",
            "components",
            "manufacturing",
            "manufacturer",
            "manufacturers",
            "shortage",
            "shortages",
        ),
    ),
)

COMPETITION_RISK = DisclosureTopic(
    key="competition_risk",
    query="competitive markets products services price competition margins",
    anchor_groups=(
        ("competitive", "competition"),
        ("product", "products", "service", "services"),
        ("price", "pricing", "margin", "margins"),
    ),
)


@dataclass(frozen=True)
class LocatedDisclosure:
    document: DocumentMetadata
    source_period: SourcePeriod
    results: tuple[VectorSearchResult, ...]
    complete_anchor_scan: bool = True


def text_matches_topic(text: str, topic: DisclosureTopic) -> bool:
    """Require at least one whole-word anchor from every topic group."""
    tokens = set(re.findall(r"[A-Za-z0-9]+", text.casefold()))
    return all(any(anchor.casefold() in tokens for anchor in group) for group in topic.anchor_groups)


def _anchor_score(text: str, topic: DisclosureTopic) -> tuple[int, int]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.casefold())
    token_set = set(tokens)
    matched_terms = {
        anchor.casefold()
        for group in topic.anchor_groups
        for anchor in group
        if anchor.casefold() in token_set
    }
    occurrences = sum(tokens.count(term) for term in matched_terms)
    return len(matched_terms), occurrences


def _normalized_disclosure(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


class DisclosureEvidenceLocator:
    """Complete anchor scan with M6 ranking for the qualifying candidates."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def locate(
        self,
        db: Session,
        *,
        document: DocumentMetadata,
        source_period: SourcePeriod,
        topic: DisclosureTopic,
        top_k: int = 2,
    ) -> LocatedDisclosure:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        chunks = db.execute(
            select(Chunk)
            .where(Chunk.document_id == document.document_id)
            .order_by(Chunk.chunk_index)
        ).scalars().all()
        anchored = [chunk for chunk in chunks if text_matches_topic(chunk.text, topic)]
        if not anchored:
            return LocatedDisclosure(
                document=document,
                source_period=source_period,
                results=(),
            )

        anchored_by_id = {chunk.id: chunk for chunk in anchored}
        ranked = self._retriever.search(
            db,
            topic.query,
            company=document.company,
            top_k=max(20, top_k),
            document_id=document.document_id,
            document_type=document.document_type,
            fiscal_year=document.fiscal_year,
        )
        ordered_ids = [
            result.chunk_id for result in ranked if result.chunk_id in anchored_by_id
        ]
        remaining = sorted(
            (chunk for chunk in anchored if chunk.id not in set(ordered_ids)),
            key=lambda chunk: (*(-value for value in _anchor_score(chunk.text, topic)), chunk.chunk_index),
        )
        ordered_ids.extend(chunk.id for chunk in remaining)
        ranked_by_id = {result.chunk_id: result for result in ranked}

        selected: list[VectorSearchResult] = []
        seen_text: set[str] = set()
        for chunk_id in ordered_ids:
            chunk = anchored_by_id[chunk_id]
            normalized = _normalized_disclosure(chunk.text)
            if normalized in seen_text:
                continue
            seen_text.add(normalized)
            selected.append(
                ranked_by_id.get(chunk_id)
                or VectorSearchResult(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    document_id=document.document_id,
                    company=document.company,
                    document_type=document.document_type,
                    fiscal_year=document.fiscal_year,
                    cosine_distance=None,
                    similarity=None,
                )
            )
            if len(selected) == top_k:
                break
        return LocatedDisclosure(
            document=document,
            source_period=source_period,
            results=tuple(selected),
        )


def _period_evidence(
    located: LocatedDisclosure,
    *,
    starting_evidence_id: int,
) -> DisclosurePeriodEvidence:
    evidence = tuple(
        TemporalEvidence(
            evidence_id=starting_evidence_id + index,
            source_period=located.source_period,
            result=result,
        )
        for index, result in enumerate(located.results)
    )
    return DisclosurePeriodEvidence(
        document_year=located.document.fiscal_year,
        source_period=located.source_period,
        evidence=evidence,
        complete_anchor_scan=located.complete_anchor_scan,
    )


def _combined_text(period: DisclosurePeriodEvidence) -> str:
    return "\n\n".join(
        f"[{item.source_period.value} filing, page {item.result.page_number}]\n"
        f"{item.result.text}"
        for item in period.evidence
    )


def classify_disclosure_change(
    *,
    older: DisclosurePeriodEvidence,
    newer: DisclosurePeriodEvidence,
    verifier: EntailmentVerifier,
) -> tuple[
    TemporalChangeCategory,
    EntailmentProbabilities | None,
    EntailmentProbabilities | None,
    str | None,
]:
    """Classify presence and bidirectional semantic entailment explicitly."""
    if not older.evidence and not newer.evidence:
        return TemporalChangeCategory.UNAVAILABLE, None, None, None
    if not older.evidence:
        category = (
            TemporalChangeCategory.NEWLY_ADDED
            if older.complete_anchor_scan
            else TemporalChangeCategory.UNAVAILABLE
        )
        return category, None, None, None
    if not newer.evidence:
        category = (
            TemporalChangeCategory.REMOVED
            if newer.complete_anchor_scan
            else TemporalChangeCategory.UNAVAILABLE
        )
        return category, None, None, None

    older_text = _combined_text(older)
    newer_text = _combined_text(newer)
    try:
        predictions = verifier.verify_many(
            [(older_text, newer_text), (newer_text, older_text)]
        )
        if len(predictions) != 2:
            raise RuntimeError("verifier returned an unexpected number of predictions")
    except Exception as exc:
        return TemporalChangeCategory.AMBIGUOUS, None, None, str(exc)

    older_entails_newer, newer_entails_older = predictions
    forward = older_entails_newer.entailment > 0.5
    reverse = newer_entails_older.entailment > 0.5
    contradiction = (
        older_entails_newer.contradiction > 0.5
        or newer_entails_older.contradiction > 0.5
    )
    if forward and reverse:
        category = TemporalChangeCategory.UNCHANGED
    elif contradiction or forward != reverse:
        category = TemporalChangeCategory.CHANGED
    else:
        category = TemporalChangeCategory.AMBIGUOUS
    return category, older_entails_newer, newer_entails_older, None


class TemporalDisclosureComparisonService:
    """Resolve, locate, and align one explicitly supported disclosure topic."""

    def __init__(
        self,
        *,
        locator: DisclosureEvidenceLocator,
        verifier: EntailmentVerifier,
        resolver: TemporalDocumentResolver | None = None,
    ) -> None:
        self._locator = locator
        self._verifier = verifier
        self._resolver = resolver or TemporalDocumentResolver()

    def compare(
        self,
        db: Session,
        *,
        company: str,
        document_type: str,
        older_year: int,
        newer_year: int,
        topic: DisclosureTopic,
        top_k: int = 2,
    ) -> DisclosureComparison:
        documents = self._resolver.resolve(
            db,
            company=company,
            document_type=document_type,
            older_year=older_year,
            newer_year=newer_year,
        )
        located_older = self._locator.locate(
            db,
            document=documents.older,
            source_period=SourcePeriod.OLDER,
            topic=topic,
            top_k=top_k,
        )
        located_newer = self._locator.locate(
            db,
            document=documents.newer,
            source_period=SourcePeriod.NEWER,
            topic=topic,
            top_k=top_k,
        )
        older = _period_evidence(located_older, starting_evidence_id=1)
        newer = _period_evidence(
            located_newer,
            starting_evidence_id=len(older.evidence) + 1,
        )
        category, forward, reverse, error = classify_disclosure_change(
            older=older,
            newer=newer,
            verifier=self._verifier,
        )
        return DisclosureComparison(
            topic=topic,
            documents=documents,
            older=older,
            newer=newer,
            category=category,
            older_entails_newer=forward,
            newer_entails_older=reverse,
            error=error,
        )
