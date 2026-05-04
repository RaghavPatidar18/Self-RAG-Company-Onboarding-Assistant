from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import redis
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

load_dotenv()


@dataclass
class CacheHit:
    answer: str
    similarity: float


class RedisSessionManager:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.ttl_seconds = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "604800"))
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _key(self, thread_id: str) -> str:
        return f"session:{thread_id}:messages"

    def append_turn(self, thread_id: str, user_message: str, assistant_message: str) -> None:
        timestamp = int(time.time())
        payloads = [
            json.dumps({"role": "user", "content": user_message, "created_at": timestamp}),
            json.dumps({"role": "assistant", "content": assistant_message, "created_at": timestamp}),
        ]
        key = self._key(thread_id)
        with self.client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *payloads)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()

    def get_messages(self, thread_id: str) -> list[dict[str, Any]]:
        raw_messages = self.client.lrange(self._key(thread_id), 0, -1)
        messages: list[dict[str, Any]] = []
        for raw in raw_messages:
            try:
                messages.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return messages

    def replace_messages(self, thread_id: str, messages: list[dict[str, Any]]) -> None:
        key = self._key(thread_id)
        serialized = [json.dumps(message) for message in messages]
        with self.client.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            if serialized:
                pipe.rpush(key, *serialized)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()

    def ping(self) -> bool:
        return bool(self.client.ping())


class RedisSemanticCache:
    def __init__(self, threshold: float = 0.88):
        self.min_similarity = threshold
        self.index_name = os.getenv("SEMANTIC_CACHE_INDEX", "idx:semantic_cache")
        self.key_prefix = os.getenv("SEMANTIC_CACHE_PREFIX", "semantic_cache:")
        self.embedding_model = os.getenv("SEMANTIC_CACHE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.ttl_seconds = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "259200"))
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        self.client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.encoder = HuggingFaceEmbeddings(model_name=self.embedding_model)
        # probe_vector = self.encoder.embed_query("semantic-cache-probe")
        self.vector_dim = 384 
        self._ensure_index()

    def _ensure_index(self) -> None:
        schema = (
            TextField("question"),
            TextField("response_text"),
            TagField("response_source"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )

        try:
            self.client.ft(self.index_name).info()
            return
        except ResponseError:
            pass

        self.client.ft(self.index_name).create_index(
            fields=schema,
            definition=IndexDefinition(prefix=[self.key_prefix], index_type=IndexType.HASH),
        )

    def _embedding_bytes(self, text: str) -> bytes:
        vector = np.asarray(self.encoder.embed_query(text), dtype=np.float32)
        return vector.tobytes()

    def get(self, query: str) -> CacheHit | None:
        vector = self._embedding_bytes(query)
        knn = Query("*=>[KNN 1 @embedding $vector AS distance]") \
            .sort_by("distance") \
            .return_fields("response_text", "distance") \
            .paging(0, 1) \
            .dialect(2)

        result = self.client.ft(self.index_name).search(knn, query_params={"vector": vector})
        if not result.docs:
            return None

        doc = result.docs[0]
        distance = float(getattr(doc, "distance", 1.0))
        similarity = max(0.0, 1.0 - distance)
        answer = getattr(doc, "response_text", b"")

        if isinstance(answer, bytes):
            answer = answer.decode("utf-8")
        if similarity < self.min_similarity or not str(answer).strip():
            return None

        return CacheHit(answer=str(answer).strip(), similarity=similarity)

    def add(self, question: str, response_text: str, response_source: str = "graph") -> None:
        answer = response_text.strip()
        if not answer:
            return

        normalized = " ".join(question.lower().split())
        key = f"{self.key_prefix}{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"
        payload = {
            "question": question,
            "response_text": answer,
            "response_source": response_source,
            "created_at": int(time.time()),
            "embedding": self._embedding_bytes(question),
        }
        with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=payload)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()
