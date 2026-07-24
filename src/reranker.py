"""
Re-ranking step: FAISS dense retrieval gives us a broad candidate set
(top_k_retrieve), then a cross-encoder scores each (query, chunk) pair
jointly for a much more precise relevance signal, and we keep only the
top_k_rerank best candidates. This is the step that most directly reduces
hallucinated answers, since the generator only ever sees genuinely
relevant context.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import CrossEncoder

from src.config import settings


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, candidates: List[dict], top_k: int = None) -> List[dict]:
    if not candidates:
        return []
    top_k = top_k or settings.top_k_rerank
    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
