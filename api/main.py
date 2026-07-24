"""
FastAPI service for the Document Q&A RAG pipeline.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /ingest        upload one or more PDFs -> OCR + chunk + index them
    POST /query         ask a question -> retrieval + rerank + generation
    GET  /evaluate       run the hand-labeled eval set -> recall@k, faithfulness
    GET  /health         liveness check
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.evaluation import run_evaluation
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Document Q&A Assistant (RAG)",
    description="Retrieval-augmented Q&A over long, multi-page PDF and scanned documents.",
    version="1.0.0",
)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    top_k_retrieve: int | None = None
    top_k_rerank: int | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "index_size": pipeline.store.index.ntotal if pipeline.store.index else 0}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    total_chunks = 0
    ingested = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} is not a PDF.")
        dest = RAW_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        n_chunks = pipeline.ingest_file(str(dest))
        total_chunks += n_chunks
        ingested.append({"filename": f.filename, "chunks_created": n_chunks})

    return {"ingested": ingested, "total_chunks_added": total_chunks}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    result = pipeline.query(req.question, req.top_k_retrieve, req.top_k_rerank)
    return result


@app.get("/evaluate")
def evaluate(top_k: int = 4):
    dataset_path = Path("data/eval/qa_dataset.json")
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="No eval dataset found at data/eval/qa_dataset.json")
    return run_evaluation(str(dataset_path), top_k=top_k)
