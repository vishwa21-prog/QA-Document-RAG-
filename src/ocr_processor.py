"""
Document ingestion & OCR preprocessing.

Handles two kinds of PDFs:
  1. "Digital" PDFs that already contain a text layer -> extracted directly
     with pdfplumber (fast, no OCR needed).
  2. Scanned / image-only PDFs -> rasterized page-by-page with pdf2image and
     run through Tesseract OCR.

A simple per-page heuristic decides which path to use: if pdfplumber
extracts fewer than `MIN_CHARS_PER_PAGE` characters from a page, we assume
it's a scanned image and fall back to OCR for that page.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

MIN_CHARS_PER_PAGE = 20  # below this, treat the page as "scanned" -> OCR


@dataclass
class PageContent:
    page_number: int          # 1-indexed
    text: str
    source: str                # "text_layer" or "ocr"
    doc_name: str


def _ocr_image(image: Image.Image) -> str:
    """Run Tesseract on a single PIL image and return cleaned text."""
    raw = pytesseract.image_to_string(image)
    return _clean_text(raw)


def _clean_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def extract_pages(pdf_path: str | Path) -> List[PageContent]:
    """
    Extract text from every page of a PDF, using the text layer when present
    and falling back to OCR for scanned pages.
    """
    pdf_path = Path(pdf_path)
    doc_name = pdf_path.name
    pages: List[PageContent] = []
    pages_needing_ocr: List[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            if len(cleaned) >= MIN_CHARS_PER_PAGE:
                pages.append(PageContent(i, cleaned, "text_layer", doc_name))
            else:
                pages.append(PageContent(i, "", "pending_ocr", doc_name))
                pages_needing_ocr.append(i)

    if pages_needing_ocr:
        logger.info(
            "%s: %d/%d pages need OCR (%s)",
            doc_name, len(pages_needing_ocr), len(pages), pages_needing_ocr,
        )
        images = convert_from_path(
            pdf_path,
            dpi=settings.ocr_dpi,
            first_page=min(pages_needing_ocr),
            last_page=max(pages_needing_ocr),
        )
        offset = min(pages_needing_ocr)
        for page_num in pages_needing_ocr:
            image = images[page_num - offset]
            text = _ocr_image(image)
            pages[page_num - 1] = PageContent(page_num, text, "ocr", doc_name)

    empty_pages = [p.page_number for p in pages if not p.text.strip()]
    if empty_pages:
        logger.warning("%s: pages with no extractable text: %s", doc_name, empty_pages)

    return pages


def extract_directory(raw_dir: str | Path) -> List[PageContent]:
    """Extract all PDFs found in a directory."""
    raw_dir = Path(raw_dir)
    all_pages: List[PageContent] = []
    for pdf_file in sorted(raw_dir.glob("*.pdf")):
        all_pages.extend(extract_pages(pdf_file))
    return all_pages
