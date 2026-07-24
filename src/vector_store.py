"""
FAISS-backed dense vector store for chunk embeddings, plus a small JSON
"docstore" that maps chunk_id -> chunk text/metadata (FAISS itself only
stores vectors + integer ids, so we keep the payload alongside it).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from src.chunking import Chunk
from src.config import settings
from src.embeddings import get_embedder


class VectorStore:
    def __init__(self, index_dir: str | Path = None, docstore_path: str | Path = None):
        self.index_dir = Path(index_dir or settings.vector_store_dir)
        self.docstore_path = Path(docstore_path or settings.doc_store_path)
        self.index: faiss.Index | None = None
        self.docstore: dict[str, dict] = {}
        self.id_to_chunk_id: dict[int, str] = {}

    # ---------------------------------------------------------- build/index
    def build(self, chunks: List[Chunk]) -> None:
        embedder = get_embedder()
        texts = [c.text for c in chunks]
        vectors = np.array(embedder.embed_documents(texts), dtype="float32")

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
        index.add(vectors)

        self.index = index
        self.docstore = {
            c.chunk_id: {
                "text": c.text,
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "source": c.source,
            }
            for c in chunks
        }
        self.id_to_chunk_id = {i: c.chunk_id for i, c in enumerate(chunks)}

    def add(self, chunks: List[Chunk]) -> None:
        """Add more chunks to an already-built index (incremental ingestion)."""
        if self.index is None:
            self.build(chunks)
            return
        embedder = get_embedder()
        texts = [c.text for c in chunks]
        vectors = np.array(embedder.embed_documents(texts), dtype="float32")
        start_id = self.index.ntotal
        self.index.add(vectors)
        for offset, c in enumerate(chunks):
            self.docstore[c.chunk_id] = {
                "text": c.text,
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "source": c.source,
            }
            self.id_to_chunk_id[start_id + offset] = c.chunk_id

    # --------------------------------------------------------------- search
    def search(self, query: str, top_k: int) -> List[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        embedder = get_embedder()
        qvec = np.array([embedder.embed_query(query)], dtype="float32")
        scores, ids = self.index.search(qvec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            chunk_id = self.id_to_chunk_id[int(idx)]
            payload = dict(self.docstore[chunk_id])
            payload["chunk_id"] = chunk_id
            payload["score"] = float(score)
            results.append(payload)
        return results

    # ------------------------------------------------------------- persist
    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "id_map.json", "w") as f:
            json.dump(self.id_to_chunk_id, f)
        self.docstore_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.docstore_path, "w") as f:
            json.dump(self.docstore, f)

    def load(self) -> None:
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "id_map.json") as f:
            self.id_to_chunk_id = {int(k): v for k, v in json.load(f).items()}
        with open(self.docstore_path) as f:
            self.docstore = json.load(f)

    def exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and self.docstore_path.exists()
