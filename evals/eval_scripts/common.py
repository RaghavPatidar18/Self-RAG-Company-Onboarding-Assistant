"""Shared deterministic Groq judge and report helpers for evaluation scripts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
MODEL = os.getenv("EVAL_GROQ_MODEL", "openai/gpt-oss-120b")


def judge(prompt: str) -> dict[str, Any]:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required (set it in .env or environment).")
    model = ChatGroq(model=MODEL, temperature=0, max_retries=2)
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
    path.write_text(json.dumps({"suite": name, "model": MODEL, "passed": passed,
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
