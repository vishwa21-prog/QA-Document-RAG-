"""
Top-level orchestrator that wires together: OCR ingestion -> chunking ->
embedding/indexing -> retrieval -> re-ranking -> generation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.chunking import chunk_pages
from src.config import settings
from src.generator import generate_answer
from src.ocr_processor import extract_directory, extract_pages
from src.reranker import rerank
from src.vector_store import VectorStore


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()
        if self.store.exists():
            self.store.load()

    # ------------------------------------------------------------- ingest
    def ingest_file(self, pdf_path: str) -> int:
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages)
        self.store.add(chunks)
        self.store.save()
        return len(chunks)

    def ingest_directory(self, raw_dir: str = "data/raw") -> int:
        pages = extract_directory(raw_dir)
        chunks = chunk_pages(pages)
        self.store.build(chunks)
        self.store.save()
        return len(chunks)

    # -------------------------------------------------------------- query
    def query(self, question: str, top_k_retrieve: int = None, top_k_rerank: int = None) -> dict:
        top_k_retrieve = top_k_retrieve or settings.top_k_retrieve
        top_k_rerank = top_k_rerank or settings.top_k_rerank

        candidates = self.store.search(question, top_k=top_k_retrieve)
        reranked = rerank(question, candidates, top_k=top_k_rerank)
        answer = generate_answer(question, reranked)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "doc_name": c["doc_name"],
                    "page_number": c["page_number"],
                    "chunk_id": c["chunk_id"],
                    "retrieval_score": c.get("score"),
                    "rerank_score": c.get("rerank_score"),
                    "text_preview": c["text"][:200],
                }
                for c in reranked
            ],
        }
