"""
Tests for PDF parsing.

We generate a tiny, known PDF at test time (via fpdf2, a test-only
dependency — see requirements-dev.txt) rather than committing a binary
sample file to the repo. This keeps the fixture's exact content
visible in the test itself, so the assertions are easy to verify by
reading the test.
"""

from pathlib import Path

import pytest
from fpdf import FPDF

from app.ingestion.parser import extract_pages

PAGE_ONE_TEXT = "Annual Report Fiscal Year 2025"
PAGE_TWO_TEXT = "Revenue increased twenty percent year over year"


@pytest.fixture
def sample_two_page_pdf(tmp_path: Path) -> Path:
    pdf = FPDF()

    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text=PAGE_ONE_TEXT)

    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text=PAGE_TWO_TEXT)

    output_path = tmp_path / "sample.pdf"
    pdf.output(str(output_path))
    return output_path


def test_extract_pages_returns_one_entry_per_page(sample_two_page_pdf: Path) -> None:
    pages = extract_pages(sample_two_page_pdf)
    assert len(pages) == 2


def test_extract_pages_preserves_page_order_and_1_indexed_numbers(
    sample_two_page_pdf: Path,
) -> None:
    pages = extract_pages(sample_two_page_pdf)

    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_extract_pages_captures_expected_text(sample_two_page_pdf: Path) -> None:
    pages = extract_pages(sample_two_page_pdf)

    assert PAGE_ONE_TEXT in pages[0].text
    assert PAGE_TWO_TEXT in pages[1].text
    # Sanity check the pages aren't cross-contaminated.
    assert PAGE_TWO_TEXT not in pages[0].text
