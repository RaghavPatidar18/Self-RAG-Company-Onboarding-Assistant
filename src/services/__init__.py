"""Service layer for ingestion, retrieval, and caching."""

from .document_reader import process_pdf, process_pptx, process_txt, read_document
from .index_docs import get_retriever, index_all_documents, index_all_documents_async
from .redis_store import RedisSemanticCache, RedisSessionManager

__all__ = [
    "RedisSemanticCache",
    "RedisSessionManager",
    "get_retriever",
    "index_all_documents",
    "index_all_documents_async",
    "process_pdf",
    "process_pptx",
    "process_txt",
    "read_document",
]
