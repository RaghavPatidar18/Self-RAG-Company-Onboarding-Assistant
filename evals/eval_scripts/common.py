"""Shared deterministic Groq judge and report helpers for evaluation scripts."""
from __future__ import annotations

import json
import atexit
import logging
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.config.logging_utils import configure_logging
from src.config.settings import settings

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
configure_logging()
MODEL = settings.eval_groq_model
logger = logging.getLogger("evals")
_EVAL_RUN_ID = uuid.uuid4().hex
_EVALUATION_RESOURCES: list[Any] = []
atexit.register(lambda: _close_evaluation_resources())


def judge(prompt: str) -> dict[str, Any]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required (set it in .env or environment).")
    model = _get_judge_model()
    rate_limit_attempts = 0
    other_attempts = 0
    while True:
        try:
            response = model.invoke(prompt)
            break
        except Exception as exc:
            is_rate_limited = getattr(exc, "status_code", None) == 429
            if is_rate_limited:
                if rate_limit_attempts >= settings.eval_rate_limit_max_retries:
                    raise
                retry_after = _retry_after_seconds(exc)
                delay = max(retry_after or 0, min(2 ** rate_limit_attempts, 60))
                rate_limit_attempts += 1
                logger.warning("Groq judge rate limited; waiting %.1fs before retry %d/%d",
                               delay, rate_limit_attempts, settings.eval_rate_limit_max_retries)
            else:
                if other_attempts >= settings.eval_groq_max_retries:
                    raise
                delay = min(2 ** other_attempts, 30)
                other_attempts += 1
                logger.warning("Groq judge request failed; retrying in %.1fs (%d/%d): %s",
                               delay, other_attempts, settings.eval_groq_max_retries, type(exc).__name__)
            time.sleep(delay)
    raw = response.content if isinstance(response.content, str) else "".join(
        str(part.get("text", "")) for part in response.content if isinstance(part, dict)
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {raw}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("pass"), bool):
        raise ValueError(f"Judge JSON must contain boolean 'pass': {raw}")
    return parsed


@lru_cache(maxsize=1)
def _get_judge_model() -> ChatGroq:
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model=MODEL,
        temperature=settings.eval_groq_temperature,
        max_retries=0,
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    match = re.search(r"try again in ([0-9.]+)s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else None


def evaluate_case(suite: str, case_id: str, evaluator: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one case independently so failures are recorded and later cases continue."""
    logger.info("Evaluation case started: %s/%s", suite, case_id)
    try:
        result = evaluator()
        result["id"] = case_id
    except Exception as exc:
        logger.exception("Evaluation case errored: %s/%s", suite, case_id)
        result = {
            "id": case_id,
            "pass": False,
            "reason": f"Evaluation execution failed ({type(exc).__name__}); see logs for details.",
            "error_type": type(exc).__name__,
        }
    if result.get("pass"):
        logger.info("Evaluation case passed: %s/%s", suite, case_id)
    else:
        logger.warning("Evaluation case failed: %s/%s: %s", suite, case_id, result.get("reason", ""))
    return result


def load_cases(name: str) -> list[dict[str, Any]]:
    return json.loads((ROOT / "evals" / "eval_dataset" / name).read_text())


def evaluation_configuration() -> dict[str, Any]:
    """Capture settings used for this run so reports remain comparable later."""
    return {
        "retrieval": {
            "chunk_size": settings.document_chunk_size,
            "chunk_overlap": settings.document_chunk_overlap,
            "top_k": settings.retriever_top_k,
            "embedding_model": settings.embedding_model,
            "qdrant_collection": settings.qdrant_collection_name,
        },
        "application": {
            "groq_model": settings.groq_model,
            "max_answer_retries": settings.max_answer_retries,
            "max_retrieval_rewrites": settings.max_retrieval_rewrites,
        },
        "judge": {
            "groq_model": settings.eval_groq_model,
            "temperature": settings.eval_groq_temperature,
            "max_retries": settings.eval_groq_max_retries,
            "rate_limit_retries": settings.eval_rate_limit_max_retries,
        },
    }


def report(name: str, results: list[dict[str, Any]]) -> int:
    passed = sum(bool(item["pass"]) for item in results)
    output = ROOT / "evals" / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%Z")
    path = output / f"{name}_{timestamp}.json"
    # Preserve every run, even if two executions begin within one second.
    suffix = 1
    while path.exists():
        path = output / f"{name}_{timestamp}_{suffix}.json"
        suffix += 1
    path.write_text(json.dumps({"suite": name, "model": MODEL,
                                "hyperparameters": evaluation_configuration(), "passed": passed,
                                "total": len(results), "results": results}, indent=2) + "\n")
    print(f"{name}: {passed}/{len(results)} passed; report: {path}")
    for row in results:
        print(f"{'PASS' if row['pass'] else 'FAIL'} {row['id']}: {row.get('reason', '')}")
    return 0 if passed == len(results) else 1


def app_answer(question: str, thread_id: str) -> str:
    """Call the same LangGraph used by the application, with isolated thread state."""
    graph = _get_evaluation_graph()
    state = graph.invoke(
        {"question": question},
        config={
            "configurable": {
                "thread_id": f"{thread_id}-{_EVAL_RUN_ID}",
                "user_id": f"evaluation-suite-{_EVAL_RUN_ID}",
            }
        },
    )
    return str(state.get("answer", ""))


@lru_cache(maxsize=1)
def _get_evaluation_graph():
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    from src.core.graph_builder import build_graph
    from src.db.base import DB_URI
    from src.services.redis_store import RedisSemanticCache, RedisSessionManager

    with psycopg.connect(DB_URI, autocommit=True) as conn:
        PostgresSaver(conn).setup()
    pool = ConnectionPool(conninfo=DB_URI, max_size=settings.postgres_pool_max_size)
    cache = RedisSemanticCache(threshold=2.0)
    sessions = RedisSessionManager()
    _EVALUATION_RESOURCES.extend([pool, cache.client, sessions.client])
    return build_graph(PostgresSaver(pool), sessions, cache)


def _close_evaluation_resources() -> None:
    from psycopg_pool import ConnectionPool

    for resource in reversed(_EVALUATION_RESOURCES):
        try:
            if isinstance(resource, ConnectionPool):
                resource.close(timeout=10.0)
            else:
                resource.close()
        except Exception:
            logger.warning("Could not close an evaluation resource cleanly", exc_info=True)
