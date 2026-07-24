"""
Lightweight unit tests that don't require downloading embedding/LLM models,
so they run fast in CI. They cover the deterministic parts of the pipeline:
chunking and evaluation metrics.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_pages
from src.ocr_processor import PageContent
from src.evaluation import recall_at_k, token_overlap_faithfulness


def test_chunk_pages_splits_long_text():
    long_text = "Sentence one. " * 200  # long enough to force multiple chunks
    pages = [PageContent(page_number=1, text=long_text, source="text_layer", doc_name="test.pdf")]
    chunks = chunk_pages(pages)
    assert len(chunks) > 1
    assert all(c.doc_name == "test.pdf" for c in chunks)
    assert all(c.page_number == 1 for c in chunks)


def test_chunk_pages_skips_empty_pages():
    pages = [
        PageContent(page_number=1, text="", source="ocr", doc_name="test.pdf"),
        PageContent(page_number=2, text="Some real content here.", source="text_layer", doc_name="test.pdf"),
    ]
    chunks = chunk_pages(pages)
    assert all(c.text.strip() for c in chunks)
    assert all(c.page_number == 2 for c in chunks)


def test_recall_at_k_hit():
    assert recall_at_k(retrieved_pages=[2, 3, 5], gold_pages=[3]) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(retrieved_pages=[1, 2], gold_pages=[9]) == 0.0


def test_recall_at_k_no_gold_pages_is_trivially_satisfied():
    assert recall_at_k(retrieved_pages=[1, 2], gold_pages=[]) == 1.0


def test_faithfulness_full_overlap():
    answer = "the sky is blue"
    contexts = ["the sky is blue today"]
    assert token_overlap_faithfulness(answer, contexts) == 1.0


def test_faithfulness_no_overlap():
    answer = "quantum entanglement theory"
    contexts = ["the sky is blue today"]
    assert token_overlap_faithfulness(answer, contexts) == 0.0


def test_faithfulness_refusal_is_faithful():
    answer = "I cannot find this in the provided document(s)."
    assert token_overlap_faithfulness(answer, []) == 1.0
