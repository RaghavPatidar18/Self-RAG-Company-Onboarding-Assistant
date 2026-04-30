import hashlib
import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
DB_PORT = os.getenv("DB_PORT")

DB_URI = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{DB_PORT}/{POSTGRES_DB}",
)

db_pool = ConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"row_factory": dict_row})


@contextmanager
def get_db_connection() -> Iterator[psycopg.Connection]:
    with db_pool.connection() as conn:
        yield conn


def init_db():
    """Initialize durable application tables."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id VARCHAR(255) PRIMARY KEY,
                    title VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Repair older databases created before activity timestamps were introduced.
            cur.execute(
                """
                ALTER TABLE chat_threads
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """
            )
            cur.execute(
                """
                ALTER TABLE chat_threads
                ADD COLUMN IF NOT EXISTS last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """
            )
            cur.execute(
                """
                UPDATE chat_threads
                SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP),
                    last_active_at = COALESCE(last_active_at, created_at, CURRENT_TIMESTAMP)
                WHERE created_at IS NULL OR last_active_at IS NULL
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGSERIAL PRIMARY KEY,
                    thread_id VARCHAR(255) REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_summaries (
                    thread_id VARCHAR(255) PRIMARY KEY REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                    summary TEXT NOT NULL DEFAULT '',
                    summarized_turns INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id BIGSERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    memory_hash VARCHAR(64) NOT NULL,
                    memory_text TEXT NOT NULL,
                    source_thread_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, memory_hash)
                )
                """
            )
        conn.commit()


def create_thread(thread_id: str, title: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_threads (thread_id, title)
                VALUES (%s, %s)
                ON CONFLICT (thread_id)
                DO UPDATE SET last_active_at = CURRENT_TIMESTAMP
                """,
                (thread_id, title),
            )
        conn.commit()


def update_thread_title(thread_id: str, title: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_threads SET title = %s, last_active_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                (title, thread_id),
            )
        conn.commit()


def get_all_threads():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thread_id, title FROM chat_threads ORDER BY last_active_at DESC, created_at DESC"
            )
            rows = cur.fetchall()
    return [(row["thread_id"], row["title"]) for row in rows]


def add_message(thread_id: str, role: str, content: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages (thread_id, role, content) VALUES (%s, %s, %s)",
                (thread_id, role, content),
            )
            cur.execute(
                "UPDATE chat_threads SET last_active_at = CURRENT_TIMESTAMP WHERE thread_id = %s",
                (thread_id,),
            )
        conn.commit()


def get_messages(thread_id: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE thread_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (thread_id,),
            )
            rows = cur.fetchall()
    return [(row["role"], row["content"]) for row in rows]


def get_thread_summary(thread_id: str) -> dict:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary, summarized_turns
                FROM thread_summaries
                WHERE thread_id = %s
                """,
                (thread_id,),
            )
            row = cur.fetchone()
    return row or {"summary": "", "summarized_turns": 0}


def upsert_thread_summary(thread_id: str, summary: str, summarized_turns: int):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO thread_summaries (thread_id, summary, summarized_turns)
                VALUES (%s, %s, %s)
                ON CONFLICT (thread_id)
                DO UPDATE SET
                    summary = EXCLUDED.summary,
                    summarized_turns = EXCLUDED.summarized_turns,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, summary, summarized_turns),
            )
        conn.commit()


def get_long_term_memories(user_id: str, limit: int = 25) -> list[str]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT memory_text
                FROM long_term_memories
                WHERE user_id = %s
                ORDER BY updated_at DESC, created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()
    return [row["memory_text"] for row in rows]


def _memory_hash(memory_text: str) -> str:
    normalized = " ".join(memory_text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def upsert_long_term_memories(user_id: str, memories: list[str], source_thread_id: str):
    cleaned_memories = [memory.strip() for memory in memories if memory and memory.strip()]
    if not cleaned_memories:
        return

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for memory in cleaned_memories:
                cur.execute(
                    """
                    INSERT INTO long_term_memories (user_id, memory_hash, memory_text, source_thread_id)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, memory_hash)
                    DO UPDATE SET
                        memory_text = EXCLUDED.memory_text,
                        source_thread_id = EXCLUDED.source_thread_id,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, _memory_hash(memory), memory, source_thread_id),
                )
        conn.commit()
