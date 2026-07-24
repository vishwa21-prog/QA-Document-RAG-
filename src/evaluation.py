"""
Evaluation harness for the RAG pipeline, run against a small hand-labeled
question set (see data/eval/qa_dataset.json).

Metrics:
  - Recall@k: for each question, did the retrieval+rerank step surface at
    least one chunk from the page(s) labeled as containing the answer,
    within the top k results?
  - Answer faithfulness: a cheap, dependency-light proxy that measures
    how much of the generated answer's content is grounded in the
    retrieved context, via token/n-gram overlap. (A cross-encoder NLI
    model can be swapped in for a stricter check -- see
    `faithfulness_nli` below -- if you want a stronger signal than
    token overlap.)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from src.rag_pipeline import RAGPipeline

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def recall_at_k(retrieved_pages: List[int], gold_pages: List[int]) -> float:
    """1.0 if any gold page appears among the retrieved pages, else 0.0."""
    if not gold_pages:
        return 1.0
    return 1.0 if set(retrieved_pages) & set(gold_pages) else 0.0


def token_overlap_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Fraction of the answer's non-trivial tokens that also appear somewhere
    in the retrieved context. 1.0 = fully grounded, 0.0 = fully unsupported.
    Ignores the pipeline's own refusal string.
    """
    if answer.strip().lower().startswith("i cannot find this"):
        return 1.0  # a correct refusal is trivially "faithful"

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for c in contexts:
        context_tokens |= _tokenize(c)
    grounded = answer_tokens & context_tokens
    return len(grounded) / len(answer_tokens)


def faithfulness_nli(answer: str, contexts: List[str]) -> float:
    """
    Optional, stricter faithfulness check using a small NLI cross-encoder
    (entailment probability of the answer given the concatenated context
    as premise). Requires `cross-encoder/nli-deberta-v3-base` to be
    downloaded; not called by default because it roughly doubles eval
    runtime -- swap it in to `run_evaluation` below if you want it.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    premise = " ".join(contexts)[:2000]
    scores = model.predict([(premise, answer)])
    # model outputs [contradiction, entailment, neutral] logits by default config;
    # treat highest-scoring label "entailment" as faithfulness proxy.
    return float(scores[0][1]) if hasattr(scores[0], "__len__") else float(scores[0])


def run_evaluation(dataset_path: str = "data/eval/qa_dataset.json", top_k: int = 4) -> dict:
    pipeline = RAGPipeline()
    dataset = json.loads(Path(dataset_path).read_text())

    per_question = []
    for item in dataset:
        question = item["question"]
        gold_pages = item.get("gold_pages", [])

        result = pipeline.query(question, top_k_rerank=top_k)
        retrieved_pages = [s["page_number"] for s in result["sources"]]
        contexts = [s["text_preview"] for s in result["sources"]]

        r_at_k = recall_at_k(retrieved_pages, gold_pages)
        faithfulness = token_overlap_faithfulness(result["answer"], contexts)

        per_question.append(
            {
                "question": question,
                "answer": result["answer"],
                "gold_pages": gold_pages,
                "retrieved_pages": retrieved_pages,
                f"recall@{top_k}": r_at_k,
                "faithfulness": faithfulness,
            }
        )

    n = len(per_question)
    summary = {
        "num_questions": n,
        f"mean_recall@{top_k}": sum(q[f"recall@{top_k}"] for q in per_question) / n if n else 0,
        "mean_faithfulness": sum(q["faithfulness"] for q in per_question) / n if n else 0,
    }
    return {"summary": summary, "per_question": per_question}
