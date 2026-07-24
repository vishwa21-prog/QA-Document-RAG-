"""
Answer generation given a question + retrieved context chunks.

Two backends are supported so the project runs end-to-end even without any
API key:
  - "openai": calls the OpenAI chat completions API (best quality).
  - "local":  runs a small local Hugging Face seq2seq model
              (google/flan-t5-base) via transformers. Lower quality, but
              fully offline and free.

The prompt explicitly instructs the model to only use the provided context
and to say so if the answer isn't contained in it -- this, combined with
re-ranking, is the main defence against hallucination.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from src.config import settings

SYSTEM_INSTRUCTIONS = (
    "You are a document question-answering assistant. Answer the question "
    "using ONLY the context passages provided below. If the answer cannot "
    "be found in the context, respond exactly with: "
    "\"I cannot find this in the provided document(s).\" "
    "Cite the page number(s) you used in parentheses, e.g. (p. 3)."
)


def _build_prompt(question: str, contexts: List[dict]) -> str:
    context_block = "\n\n".join(
        f"[Passage {i+1} | {c['doc_name']} p.{c['page_number']}]\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


def _generate_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


@lru_cache(maxsize=1)
def _local_pipeline():
    from transformers import pipeline

    return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)


def _generate_local(prompt: str) -> str:
    pipe = _local_pipeline()
    out = pipe(prompt, do_sample=False)
    return out[0]["generated_text"].strip()


def generate_answer(question: str, contexts: List[dict]) -> str:
    if not contexts:
        return "I cannot find this in the provided document(s)."
    prompt = _build_prompt(question, contexts)
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return _generate_openai(prompt)
    return _generate_local(prompt)
