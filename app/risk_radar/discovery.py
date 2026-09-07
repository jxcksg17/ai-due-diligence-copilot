"""Repeatable Item 1A risk-evidence discovery and deterministic signaling."""

import re
from dataclasses import replace

from sqlalchemy.orm import Session

from app.retrieval.base import Retriever
from app.retrieval.metadata import DocumentMetadata
from app.retrieval.vector_store import VectorSearchResult
from app.risk_radar.models import (
    RiskEvidence,
    RiskPresenceState,
    RiskSignal,
)
from app.risk_radar.section import Item1ASectionLocator, RiskSectionChunk
from app.risk_radar.taxonomy import RiskTopic
from app.temporal.models import SourcePeriod


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def text_matches_risk_topic(text: str, topic: RiskTopic) -> bool:
    """Require a whole-word match from every explicit topic anchor group."""
    tokens = {token.casefold() for token in _TOKEN_PATTERN.findall(text)}
    return all(
        any(anchor.casefold() in tokens for anchor in group)
        for group in topic.anchor_groups
    )


def _anchor_score(text: str, topic: RiskTopic) -> tuple[int, int]:
    tokens = [token.casefold() for token in _TOKEN_PATTERN.findall(text)]
    token_set = set(tokens)
    matched = {
        anchor.casefold()
        for group in topic.anchor_groups
        for anchor in group
        if anchor.casefold() in token_set
    }
    return len(matched), sum(tokens.count(anchor) for anchor in matched)


def _normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


class RiskEvidenceDiscovery:
    """Scan Item 1A completely, then rank only deterministic topic matches."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        section_locator: Item1ASectionLocator | None = None,
    ) -> None:
        self._retriever = retriever
        self._section_locator = section_locator or Item1ASectionLocator()

    def discover(
        self,
        db: Session,
        *,
        document: DocumentMetadata,
        source_period: SourcePeriod,
        topic: RiskTopic,
        top_k: int = 3,
        starting_evidence_id: int = 1,
    ) -> RiskSignal:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if starting_evidence_id < 1:
            raise ValueError("starting evidence ID must be at least 1")

        section = self._section_locator.locate(db, document)
        matched = [
            chunk
            for chunk in section.chunks
            if text_matches_risk_topic(chunk.text, topic)
        ]
        unique_by_text: dict[str, RiskSectionChunk] = {}
        for chunk in matched:
            unique_by_text.setdefault(_normalized_text(chunk.text), chunk)
        unique_chunks = list(unique_by_text.values())
        if not unique_chunks:
            return RiskSignal(
                topic=topic,
                document=document,
                source_period=source_period,
                presence=RiskPresenceState.NOT_FOUND,
                evidence=(),
                qualifying_passage_count=0,
                complete_item_1a_scan=section.complete,
            )

        candidate_by_id = {chunk.chunk_id: chunk for chunk in unique_chunks}
        ranked = self._retriever.search(
            db,
            topic.query,
            company=document.company,
            top_k=max(20, top_k),
            document_id=document.document_id,
            document_type=document.document_type,
            fiscal_year=document.fiscal_year,
        )
        ranked = [
            result
            for result in ranked
            if result.chunk_id in candidate_by_id
            and result.document_id == document.document_id
        ]
        ordered_ids = [result.chunk_id for result in ranked]
        ranked_ids = set(ordered_ids)
        remaining = sorted(
            (
                chunk
                for chunk in unique_chunks
                if chunk.chunk_id not in ranked_ids
            ),
            key=lambda chunk: (
                *_descending(_anchor_score(chunk.text, topic)),
                chunk.chunk_index,
            ),
        )
        ordered_ids.extend(chunk.chunk_id for chunk in remaining)
        ranked_by_id = {result.chunk_id: result for result in ranked}

        selected: list[RiskEvidence] = []
        for chunk_id in ordered_ids[:top_k]:
            chunk = candidate_by_id[chunk_id]
            ranked_result = ranked_by_id.get(chunk_id)
            result = (
                replace(
                    ranked_result,
                    text=chunk.text,
                    page_number=chunk.page_number,
                )
                if ranked_result is not None
                else VectorSearchResult(
                    chunk_id=chunk.chunk_id,
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
            selected.append(
                RiskEvidence(
                    evidence_id=starting_evidence_id + len(selected),
                    topic_key=topic.key,
                    source_period=source_period,
                    result=result,
                )
            )

        return RiskSignal(
            topic=topic,
            document=document,
            source_period=source_period,
            presence=RiskPresenceState.DISCLOSED,
            evidence=tuple(selected),
            qualifying_passage_count=len(unique_chunks),
            complete_item_1a_scan=section.complete,
        )


def _descending(values: tuple[int, int]) -> tuple[int, int]:
    return tuple(-value for value in values)
