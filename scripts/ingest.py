#!/usr/bin/env python
"""
CLI helper to (re)build the FAISS index from every PDF in data/raw/.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --raw-dir data/raw
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag_pipeline import RAGPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing source PDFs.")
    args = parser.parse_args()

    pdfs = list(Path(args.raw_dir).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {args.raw_dir}. Drop some in there first.")
        return

    print(f"Found {len(pdfs)} PDF(s). Extracting text (OCR fallback for scanned pages)...")
    start = time.time()

    pipeline = RAGPipeline()
    n_chunks = pipeline.ingest_directory(args.raw_dir)

    print(f"Indexed {n_chunks} chunks from {len(pdfs)} document(s) in {time.time() - start:.1f}s")
    print(f"Vector store saved to: data/processed/")


if __name__ == "__main__":
    main()
