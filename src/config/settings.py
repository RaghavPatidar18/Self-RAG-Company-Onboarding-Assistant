from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _resolve_path(raw_value: str | None, default: str) -> Path:
    path = Path(raw_value or default)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


class Settings:
    project_root = BASE_DIR
    docs_dir = _resolve_path(os.getenv("DOCS_FOLDER"), "documents")

    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    postgres_user = os.getenv("POSTGRES_USER", "rag_user")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "rag_password")
    postgres_db = os.getenv("POSTGRES_DB", "rag_db")
    db_port = os.getenv("DB_PORT", "5432")
    database_url = os.getenv(
        "DATABASE_URL",
        f"postgresql://{postgres_user}:{postgres_password}@localhost:{db_port}/{postgres_db}",
    )

    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "ocr_chunks")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_vector_dimension = int(os.getenv("QDRANT_VECTOR_DIMENSION", "384"))
    qdrant_upsert_batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64"))
    retriever_top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
    document_chunk_size = int(os.getenv("DOCUMENT_CHUNK_SIZE", "512"))
    document_chunk_overlap = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "200"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    semantic_cache_threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.88"))
    semantic_cache_index = os.getenv("SEMANTIC_CACHE_INDEX", "idx:semantic_cache")
    semantic_cache_prefix = os.getenv("SEMANTIC_CACHE_PREFIX", "semantic_cache:")
    semantic_cache_embedding_model = os.getenv("SEMANTIC_CACHE_EMBEDDING_MODEL", embedding_model)
    semantic_cache_ttl_seconds = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "259200"))
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_session_ttl_seconds = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "604800"))

    short_term_max_turns = int(os.getenv("SHORT_TERM_MAX_TURNS", "10"))
    short_term_retain_turns = int(os.getenv("SHORT_TERM_RETAIN_TURNS", "4"))
    max_answer_retries = int(os.getenv("MAX_ANSWER_RETRIES", "3"))
    max_retrieval_rewrites = int(os.getenv("MAX_RETRIEVAL_REWRITES", "3"))
    postgres_pool_max_size = int(os.getenv("POSTGRES_POOL_MAX_SIZE", "20"))
    eval_groq_model = os.getenv("EVAL_GROQ_MODEL", groq_model)
    eval_groq_temperature = float(os.getenv("EVAL_GROQ_TEMPERATURE", "0"))
    eval_groq_max_retries = int(os.getenv("EVAL_GROQ_MAX_RETRIES", "5"))
    eval_rate_limit_max_retries = int(os.getenv("EVAL_RATE_LIMIT_MAX_RETRIES", "8"))

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE", "logs/app.log")
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "3"))


settings = Settings()

__all__ = ["settings", "BASE_DIR"]
