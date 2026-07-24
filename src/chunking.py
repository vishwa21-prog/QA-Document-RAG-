"""
Splits extracted page text into overlapping chunks suitable for embedding.

Uses LangChain's RecursiveCharacterTextSplitter, which tries progressively
finer separators (paragraph -> sentence -> word) so chunks break on natural
boundaries rather than mid-sentence whenever possible.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings
from src.ocr_processor import PageContent


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_number: int
    text: str
    source: str  # "text_layer" or "ocr"
    metadata: dict = field(default_factory=dict)


def _make_chunk_id(doc_name: str, page_number: int, chunk_index: int) -> str:
    raw = f"{doc_name}:{page_number}:{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def chunk_pages(pages: List[PageContent]) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        pieces = splitter.split_text(page.text)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(page.doc_name, page.page_number, idx),
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    text=piece,
                    source=page.source,
                    metadata={
                        "doc_name": page.doc_name,
                        "page_number": page.page_number,
                        "source": page.source,
                    },
                )
            )
    return chunks
