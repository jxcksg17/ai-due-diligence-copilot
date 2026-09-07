"""Deterministic Item 1A section location over page-aware filing chunks."""

import re
from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.retrieval.metadata import DocumentMetadata


class RiskSectionError(ValueError):
    """Raised when a complete and unique Item 1A span cannot be established."""


@dataclass(frozen=True)
class RiskSectionChunk:
    chunk_id: int
    chunk_index: int
    page_number: int
    text: str


@dataclass(frozen=True)
class Item1ASection:
    document: DocumentMetadata
    chunks: tuple[RiskSectionChunk, ...]
    complete: bool = True


_ITEM_1A_PATTERN = re.compile(
    r"\bitem\s+1a\s*[.:-]\s*risk\s+factors\b",
    re.IGNORECASE,
)
_ITEM_1B_PATTERN = re.compile(r"\bitem\s+1b\s*[.:-]", re.IGNORECASE)


def extract_item_1a_section(
    chunks: Sequence[Any],
    *,
    document: DocumentMetadata,
) -> Item1ASection:
    """Clip the filing's actual Item 1A span, excluding TOC and Item 1B text."""
    start_candidates: list[tuple[int, re.Match[str]]] = []
    for index, chunk in enumerate(chunks):
        match = _ITEM_1A_PATTERN.search(chunk.text)
        if match is not None and "table of contents" not in chunk.text.casefold():
            start_candidates.append((index, match))
    if not start_candidates:
        raise RiskSectionError("Item 1A - Risk Factors section was not found")
    if len(start_candidates) > 1:
        raise RiskSectionError("Item 1A - Risk Factors section is ambiguous")

    start_index, start_match = start_candidates[0]
    end_index: int | None = None
    end_match: re.Match[str] | None = None
    for index in range(start_index + 1, len(chunks)):
        match = _ITEM_1B_PATTERN.search(chunks[index].text)
        if match is not None:
            end_index = index
            end_match = match
            break
    if end_index is None or end_match is None:
        raise RiskSectionError("Item 1A end boundary at Item 1B was not found")

    section_chunks: list[RiskSectionChunk] = []
    for index in range(start_index, end_index + 1):
        chunk = chunks[index]
        text = chunk.text
        if index == start_index:
            text = text[start_match.start() :]
        if index == end_index:
            text = text[: end_match.start()]
        if text.strip():
            section_chunks.append(
                RiskSectionChunk(
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    text=text.strip(),
                )
            )
    if not section_chunks:
        raise RiskSectionError("Item 1A section did not contain usable text")
    return Item1ASection(document=document, chunks=tuple(section_chunks))


class Item1ASectionLocator:
    """Database-backed adapter for deterministic section extraction."""

    def locate(self, db: Session, document: DocumentMetadata) -> Item1ASection:
        chunks = db.execute(
            select(Chunk)
            .where(Chunk.document_id == document.document_id)
            .order_by(Chunk.chunk_index)
        ).scalars().all()
        return extract_item_1a_section(chunks, document=document)
