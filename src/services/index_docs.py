from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from rich.console import Console

from src.config.settings import settings
from src.services.document_reader import process_pdf, process_pptx, process_txt

load_dotenv()

console = Console()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _build_async_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url)


def _build_sync_qdrant_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


async def _recreate_collection(client: AsyncQdrantClient, vector_size: int) -> bool:
    try:
        if await client.collection_exists(settings.qdrant_collection_name):
            await client.delete_collection(settings.qdrant_collection_name)

        await client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        return True
    except Exception as exc:  # pragma: no cover
        console.print(f"[bold red]Failed to recreate Qdrant collection:[/] {exc}")
        return False


def _chunked(items: list[Document], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _serialize_payload_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_payload_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_payload_value(item) for item in value]
    return str(value)


def _build_point_payload(doc: Document) -> dict[str, Any]:
    metadata = _serialize_payload_value(doc.metadata) if doc.metadata else {}
    source = metadata.get("source") if isinstance(metadata, dict) else ""
    return {
        "page_content": doc.page_content,
        "source": str(source or ""),
        "metadata": metadata,
    }


async def _upsert_chunks(chunks: list[Document], batch_size: int = settings.qdrant_upsert_batch_size) -> int:
    if not chunks:
        return 0

    client = _build_async_qdrant_client()
    try:
        probe_vector = await asyncio.to_thread(embeddings.embed_query, "embedding-dimension-probe")
        vector_size = len(probe_vector) or settings.qdrant_vector_dimension
        ok = await _recreate_collection(client, vector_size=vector_size)
        if not ok:
            return 0

        upserted = 0
        for batch in _chunked(chunks, batch_size=batch_size):
            batch_vectors = await asyncio.to_thread(
                embeddings.embed_documents,
                [doc.page_content for doc in batch],
            )
            points = [
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=_build_point_payload(doc),
                )
                for doc, vector in zip(batch, batch_vectors)
            ]
            await client.upsert(
                collection_name=settings.qdrant_collection_name,
                points=points,
                wait=True,
            )
            upserted += len(points)
        return upserted
    finally:
        await client.close()


def _ensure_collection_exists(client: QdrantClient) -> bool:
    try:
        return client.collection_exists(settings.qdrant_collection_name)
    except Exception as exc:  # pragma: no cover
        console.print(f"[bold red]Failed to check Qdrant collection:[/] {exc}")
        return False


def _query_qdrant(client: QdrantClient, query: str, limit: int) -> list[Document]:
    query_vector = embeddings.embed_query(query)
    query_response = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    documents: list[Document] = []
    for point in query_response.points:
        payload = point.payload or {}
        page_content = str(payload.get("page_content", "")).strip()
        if not page_content:
            continue

        raw_metadata = payload.get("metadata", {})
        if isinstance(raw_metadata, dict):
            metadata: dict[str, Any] = dict(raw_metadata)
        else:
            metadata = {}

        if not metadata:
            metadata = {k: v for k, v in payload.items() if k != "page_content"}

        source = payload.get("source")
        if source and "source" not in metadata:
            metadata["source"] = source
        metadata["score"] = point.score

        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


class _QdrantRetriever:
    def __init__(self, client: QdrantClient, k: int):
        self._client = client
        self._k = k

    def invoke(self, query: str) -> list[Document]:
        return _query_qdrant(client=self._client, query=query, limit=self._k)


def text_to_documents(text: str, filename: str, file_type: str) -> list[Document]:
    docs: list[Document] = []

    if file_type == "pdf":
        parts = re.split(r"--- Page (\d+) ---", text)
        for i in range(1, len(parts), 2):
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename, "file_type": file_type, "page": int(parts[i])},
                    )
                )

    elif file_type == "pptx":
        parts = re.split(r"--- Slide (\d+) ---", text)
        for i in range(1, len(parts), 2):
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename, "file_type": file_type, "slide": int(parts[i])},
                    )
                )

    else:
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": filename, "file_type": file_type}))

    if not docs and text.strip():
        docs.append(Document(page_content=text, metadata={"source": filename, "file_type": file_type}))

    return docs


async def index_all_documents_async() -> int:
    docs_dir = settings.docs_dir
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)

    all_docs: list[Document] = []
    for filename in sorted(os.listdir(docs_dir)):
        path = docs_dir / filename
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                console.print(f"\n[bold cyan]Processing PDF:[/] {filename}")
                text = process_pdf(str(path))
                docs = text_to_documents(text, filename, "pdf")
            elif ext == ".txt":
                console.print(f"\n[bold cyan]Processing TXT:[/] {filename}")
                text = process_txt(str(path))
                docs = text_to_documents(text, filename, "txt")
            elif ext == ".pptx":
                console.print(f"\n[bold cyan]Processing PPTX:[/] {filename}")
                text = process_pptx(str(path))
                docs = text_to_documents(text, filename, "pptx")
            else:
                continue

            all_docs.extend(docs)
            console.print(f"[bold green]✓[/] Extracted {len(docs)} section(s) from {filename}")
        except Exception as exc:
            console.print(f"[bold red]Error processing {filename}:[/] {exc}")
            continue

    if not all_docs:
        console.print("[bold yellow]No documents found to index.[/]")
        return 0

    console.print(f"\n[bold]Chunking {len(all_docs)} section(s) into smaller pieces...[/]")
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    with console.status("[bold green]Upserting chunks into Qdrant..."):
        upserted = await _upsert_chunks(chunks)
    return upserted


def index_all_documents() -> int:
    return asyncio.run(index_all_documents_async())


def get_retriever(k: int = 4):
    client = _build_sync_qdrant_client()
    try:
        if not _ensure_collection_exists(client):
            return None
        return _QdrantRetriever(client=client, k=k)
    except Exception as exc:  # pragma: no cover
        console.print(f"[bold red]Failed to initialize Qdrant retriever:[/] {exc}")
        return None


if __name__ == "__main__":
    total = index_all_documents()
    console.print(f"\n[bold]Total chunks indexed:[/] {total}")
