"""
PDF parsing: extracts per-page plain text.

Why pypdf: lightweight, pure-Python, no system dependencies (unlike
some layout/table-extraction libraries that shell out to Ghostscript
or a JVM). Good enough for continuous-prose documents like annual
reports.

Known, accepted limitation for M2 (tracked as a risk in the
architecture document): pypdf does not reliably reconstruct tables or
multi-column layouts — a table's cells can come out as jumbled text.
A dedicated layout-aware parser is a candidate for a later milestone,
once we've actually observed this failing on a real financial
document, rather than being built speculatively now.
"""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-indexed, matching how a citation refers to "page 12"
    text: str


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract plain text from every page of a PDF, in order."""
    reader = PdfReader(str(pdf_path))

    pages: list[PageText] = []
    for zero_indexed_position, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        pages.append(PageText(page_number=zero_indexed_position + 1, text=raw_text.strip()))
    return pages
