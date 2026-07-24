"""
Central configuration for the Document Q&A RAG pipeline.
All values can be overridden via environment variables or a `.env` file
(see .env.example).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "local"          # "openai" or "local"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Re-ranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # OCR
    tesseract_cmd: str = "tesseract"
    ocr_dpi: int = 300

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k_retrieve: int = 20
    top_k_rerank: int = 4

    # Storage
    vector_store_dir: str = "data/processed/faiss_index"
    doc_store_path: str = "data/processed/docstore.json"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
