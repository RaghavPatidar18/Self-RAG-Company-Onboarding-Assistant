import os
import time
from uuid import uuid4

from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models


class SemanticCache:
    def __init__(self, threshold: float = 0.88):
        self.threshold = threshold
        self.collection_name = os.getenv("SEMANTIC_CACHE_COLLECTION", "semantic_cache")
        self.embedding_model = os.getenv("SEMANTIC_CACHE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        cache_path = os.getenv("SEMANTIC_CACHE_PATH", ":memory:")
        if cache_path == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=cache_path)

        self.encoder = HuggingFaceEmbeddings(model_name=self.embedding_model)
        probe_vector = self.encoder.embed_query("semantic-cache-probe")
        self.vector_size = len(probe_vector)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _get_embedding(self, text: str) -> list[float]:
        return self.encoder.embed_query(text)

    def get(self, query: str) -> tuple[str, float] | None:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=self._get_embedding(query),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not result.points:
            return None

        hit = result.points[0]
        score = float(hit.score or 0.0)
        if score < self.threshold:
            return None

        payload = hit.payload or {}
        response_text = str(payload.get("response_text", "")).strip()
        if not response_text:
            return None
        return response_text, score

    def add(self, question: str, response_text: str) -> None:
        clean_response = response_text.strip()
        if not clean_response:
            return

        point = models.PointStruct(
            id=str(uuid4()),
            vector=self._get_embedding(question),
            payload={
                "question": question,
                "response_text": clean_response,
                "created_at": int(time.time()),
            },
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
            wait=True,
        )
