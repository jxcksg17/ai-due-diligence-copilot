"""
Unit tests for chunking. No database, no PDF, no I/O — these test pure
functions and should run in milliseconds.
"""

from app.ingestion.chunking import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    chunk_document,
    chunk_page_text,
)
from app.ingestion.parser import PageText


def test_empty_page_text_produces_no_chunks() -> None:
    assert chunk_page_text(page_number=1, text="") == []


def test_short_text_produces_a_single_chunk() -> None:
    text = "Revenue increased in the current fiscal year."
    chunks = chunk_page_text(page_number=5, text=text)

    assert len(chunks) == 1
    assert chunks[0].page_number == 5
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == text


def test_long_text_is_split_into_multiple_overlapping_chunks() -> None:
    # Build text longer than one chunk so we can verify splitting behavior.
    text = "A" * (CHUNK_SIZE_CHARS + 500)

    chunks = chunk_page_text(page_number=2, text=text)

    assert len(chunks) >= 2
    # Every chunk must report the same page number it came from.
    assert all(c.page_number == 2 for c in chunks)
    # chunk_index must increase without gaps or repeats.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # No chunk should exceed the configured max size.
    assert all(len(c.text) <= CHUNK_SIZE_CHARS for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = "B" * (CHUNK_SIZE_CHARS + 400)

    chunks = chunk_page_text(page_number=1, text=text)

    first_chunk_tail = chunks[0].text[-CHUNK_OVERLAP_CHARS:]
    second_chunk_head = chunks[1].text[:CHUNK_OVERLAP_CHARS]
    assert first_chunk_tail == second_chunk_head


def test_chunk_document_uses_one_continuous_index_across_pages() -> None:
    pages = [
        PageText(page_number=1, text="Short page one text."),
        PageText(page_number=2, text="Short page two text."),
        PageText(page_number=3, text="Short page three text."),
    ]

    chunks = chunk_document(pages)

    assert len(chunks) == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert [c.page_number for c in chunks] == [1, 2, 3]


def test_chunk_document_skips_blank_pages() -> None:
    pages = [
        PageText(page_number=1, text="Has content."),
        PageText(page_number=2, text=""),  # e.g. a blank/image-only page
        PageText(page_number=3, text="Has content too."),
    ]

    chunks = chunk_document(pages)

    assert len(chunks) == 2
    assert [c.page_number for c in chunks] == [1, 3]
