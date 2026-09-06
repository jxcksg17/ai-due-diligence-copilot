"""
Naive fixed-size chunking.

Two deliberate constraints, explained here because they shape every
chunk this module produces:

1. Chunks never span a page boundary. `page_number` is core citation
   metadata ("Annual Report 2025, page 82"), and letting a chunk
   straddle two pages would make that metadata ambiguous. The cost is
   that a sentence split across a page break can end up separated
   across two chunks — an acceptable trade-off for a naive first pass.
   Metadata-aware / semantic chunking (a later milestone) can improve
   on this once it matters.

2. Chunking is character-based, not token-based. This avoids adding a
   tokenizer dependency before we've even chosen an embedding model.
   Token-aligned chunking (sized to the embedding model's actual
   context window) is a natural improvement once M3 picks that model.
"""

from dataclasses import dataclass

from app.ingestion.parser import PageText

CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_index: int
    text: str


def chunk_page_text(page_number: int, text: str, start_index: int = 0) -> list[TextChunk]:
    """Split one page's text into fixed-size, overlapping chunks."""
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = start_index
    text_length = len(text)

    while start < text_length:
        end = min(start + CHUNK_SIZE_CHARS, text_length)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                TextChunk(page_number=page_number, chunk_index=index, text=chunk_text)
            )
            index += 1

        if end == text_length:
            break
        # Step back by the overlap amount so consecutive chunks share
        # some context, which helps retrieval later when a relevant
        # sentence happens to sit right at a chunk boundary.
        start = end - CHUNK_OVERLAP_CHARS

    return chunks


def chunk_document(pages: list[PageText]) -> list[TextChunk]:
    """Chunk every page of a document, with one continuous chunk_index
    across the whole document (not reset per page) so chunks can be
    reassembled in reading order.
    """
    all_chunks: list[TextChunk] = []
    next_index = 0
    for page in pages:
        page_chunks = chunk_page_text(page.page_number, page.text, start_index=next_index)
        all_chunks.extend(page_chunks)
        next_index += len(page_chunks)
    return all_chunks
