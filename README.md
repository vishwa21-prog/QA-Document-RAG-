# Document Q&A Assistant (RAG)

A retrieval-augmented generation (RAG) pipeline that answers questions over long, multi-page PDF and scanned documents — combining OCR preprocessing with dense embeddings, chunking, vector search, and a re-ranking step to reduce hallucinated answers. Retrieval quality is evaluated with recall@k and answer faithfulness against a hand-labeled question set.

**Tech stack:** Python, LangChain, FAISS, Hugging Face Transformers (sentence-transformers + cross-encoders), Tesseract OCR, FastAPI.

---

## How it works

```
PDF(s) in data/raw/
      │
      ▼
┌─────────────────┐   digital PDF → pdfplumber text layer
│  OCR Processor   │   scanned PDF → pdf2image + Tesseract OCR
└─────────────────┘   (per-page fallback, see src/ocr_processor.py)
      │
      ▼
┌─────────────────┐   RecursiveCharacterTextSplitter
│    Chunking      │   (paragraph → sentence → word boundaries)
└─────────────────┘
      │
      ▼
┌─────────────────┐   sentence-transformers/all-MiniLM-L6-v2
│   Embeddings     │   (normalized, cosine similarity)
└─────────────────┘
      │
      ▼
┌─────────────────┐   FAISS IndexFlatIP + JSON docstore
│  Vector Store    │   (data/processed/)
└─────────────────┘
      │  top_k_retrieve candidates
      ▼
┌─────────────────┐   cross-encoder/ms-marco-MiniLM-L-6-v2
│   Re-ranker      │   scores (query, chunk) pairs jointly
└─────────────────┘
      │  top_k_rerank best chunks
      ▼
┌─────────────────┐   OpenAI or local flan-t5-base
│   Generator      │   answers ONLY from provided context,
└─────────────────┘   refuses if the answer isn't there
      │
      ▼
   Answer + cited sources (doc name, page number, scores)
```

The re-ranking step is the main lever against hallucination: dense retrieval alone (top_k=20) is optimized for recall, so it often surfaces topically-related-but-irrelevant chunks. The cross-encoder re-scores each candidate jointly with the query (rather than via separately-computed embeddings), which is slower per-pair but far more precise, so only the genuinely relevant handful of chunks ever reach the LLM.

## Project structure

```
document-qa-rag/
├── src/
│   ├── config.py           # central settings (env-driven)
│   ├── ocr_processor.py    # PDF text extraction + Tesseract OCR fallback
│   ├── chunking.py         # text splitting into overlapping chunks
│   ├── embeddings.py       # HF embedding model wrapper
│   ├── vector_store.py     # FAISS index + JSON docstore
│   ├── reranker.py         # cross-encoder re-ranking
│   ├── generator.py        # LLM answer generation (OpenAI or local)
│   ├── rag_pipeline.py     # orchestrates the full pipeline
│   └── evaluation.py       # recall@k + faithfulness metrics
├── api/
│   └── main.py             # FastAPI app: /ingest /query /evaluate /health
├── scripts/
│   ├── ingest.py            # CLI: build the index from data/raw/*.pdf
│   └── evaluate.py          # CLI: run the eval harness
├── data/
│   ├── raw/                 # drop your source PDFs here
│   ├── processed/           # FAISS index + docstore get written here
│   └── eval/qa_dataset.json # hand-labeled question set for evaluation
├── tests/
│   └── test_pipeline.py     # unit tests (no model downloads required)
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Setup

```bash
git clone <your-repo-url>
cd document-qa-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

You also need Tesseract OCR and Poppler installed on the system (not just the Python bindings):

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils
```

By default the project runs fully offline with a small local model (`google/flan-t5-base`) for generation, so you can try it with zero API keys. For better answer quality, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...` in `.env`.

## Usage

### 1. Add documents and build the index

Drop one or more PDFs (digital or scanned) into `data/raw/`, then:

```bash
python scripts/ingest.py
```

This extracts text (OCR'ing scanned pages automatically), chunks it, embeds every chunk, and writes a FAISS index + docstore to `data/processed/`.

### 2. Ask questions via the API

```bash
uvicorn api.main:app --reload --port 8000
```

Then, in another terminal:

```bash
# Upload + ingest a PDF via the API instead of the CLI script
curl -X POST http://localhost:8000/ingest -F "files=@/path/to/your.pdf"

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination notice period?"}'
```

Interactive API docs are auto-generated at `http://localhost:8000/docs`.

### 3. Evaluate retrieval + faithfulness

Edit `data/eval/qa_dataset.json` with real questions and the page number(s) where each answer actually appears in your ingested PDFs, then:

```bash
python scripts/evaluate.py --verbose
```

This reports:
- **recall@k** — did retrieval + re-ranking surface a chunk from the correct page within the top k results?
- **faithfulness** — how much of the generated answer's content is actually grounded in the retrieved context (token-overlap proxy; a stricter NLI-based variant is included in `src/evaluation.py::faithfulness_nli` if you want a stronger signal).

### 4. Run unit tests

```bash
pytest tests/ -v
```

### 5. Run with Docker

```bash
docker build -t document-qa-rag .
docker run -p 8000:8000 -v $(pwd)/data:/app/data document-qa-rag
```

## Design notes / things to highlight when discussing this project

- **Per-page OCR fallback rather than whole-document**: each page is checked individually for an extractable text layer, so a mostly-digital PDF with a few scanned pages only pays the (slow) OCR cost where it's actually needed.
- **Recursive chunking**: splits on paragraph → sentence → word boundaries in that order, which keeps chunks semantically coherent instead of cutting mid-sentence.
- **Cosine similarity via normalized embeddings + `IndexFlatIP`**: simplest correct way to do cosine search in FAISS without maintaining a separate normalization step at query time.
- **Two-stage retrieval (dense retrieval → cross-encoder re-rank)**: a standard and effective pattern for balancing recall (cheap, approximate, over many candidates) against precision (expensive, exact, over few candidates).
- **Hallucination mitigation** happens at three points: (1) re-ranking removes irrelevant context before it ever reaches the LLM, (2) the generation prompt explicitly instructs the model to answer only from context and refuse otherwise, (3) the faithfulness metric gives you a measurable way to catch regressions.
- **Swappable LLM backend**: local flan-t5 for offline/free iteration, OpenAI for production-quality answers — controlled entirely via `.env`, no code changes.

## Possible extensions

- Add a BM25 (keyword) retriever alongside FAISS and combine scores (hybrid search) — `rank-bm25` is already in `requirements.txt` for this.
- Swap `token_overlap_faithfulness` for `faithfulness_nli` (already implemented) for a stricter, model-based groundedness check.
- Add streaming responses from `/query` for long answers.
- Add per-document access control / multi-tenant docstores if used for confidential documents.
