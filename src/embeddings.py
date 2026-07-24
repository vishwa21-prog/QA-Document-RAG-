"""
Thin wrapper around a Hugging Face sentence-transformers model, used both
for building the FAISS index and for embedding queries at retrieval time.
Kept as a small singleton-style accessor so the (fairly large) model is
loaded into memory only once per process.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
