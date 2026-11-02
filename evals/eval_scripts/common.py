"""Shared deterministic Groq judge and report helpers for evaluation scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.config.settings import settings

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
MODEL = settings.eval_groq_model


def judge(prompt: str) -> dict[str, Any]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required (set it in .env or environment).")
    model = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model=MODEL,
        temperature=settings.eval_groq_temperature,
        max_retries=settings.eval_groq_max_retries,
    )
    response = model.invoke(prompt)
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
    from src.core.graph_builder import build_graph
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from src.config.settings import settings
    from src.db.base import DB_URI
    from src.services.redis_store import RedisSessionManager, RedisSemanticCache

    with psycopg.connect(DB_URI, autocommit=True) as conn:
        PostgresSaver(conn).setup()
    cache = RedisSemanticCache(threshold=1.0)
    sessions = RedisSessionManager()
    with psycopg.connect(DB_URI) as conn:
        graph = build_graph(PostgresSaver(conn), sessions, cache)
        state = graph.invoke(
            {"question": question},
            config={"configurable": {"thread_id": thread_id, "user_id": "evaluation-suite"}},
        )
    return str(state.get("answer", ""))
