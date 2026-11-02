"""Streamlit dashboard for timestamped evaluation reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config.settings import BASE_DIR

RESULTS_DIR = BASE_DIR / "evals" / "results"


def _load_runs() -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    suites: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    if not RESULTS_DIR.exists():
        return suites

    for suite_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
        runs = []
        for report_path in suite_dir.glob("*.json"):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(report, dict) and isinstance(report.get("results"), list):
                    runs.append((report_path, report))
            except (OSError, json.JSONDecodeError):
                continue
        if runs:
            suites[suite_dir.name] = sorted(runs, key=lambda row: row[0].stat().st_mtime)
    return suites


def _rate(report: dict[str, Any]) -> float:
    total = int(report.get("total", len(report.get("results", []))))
    passed = int(report.get("passed", sum(bool(row.get("pass")) for row in report.get("results", []))))
    return passed / total * 100 if total else 0.0


def render() -> None:
    st.title("Evaluation Dashboard")
    st.caption("Review LLM-as-judge results saved by the evaluation scripts.")
    suites = _load_runs()

    if not suites:
        st.info(
            "No evaluation reports found yet. Run an evaluation suite from the repository root; "
            "reports will appear here automatically."
        )
        st.code("python -m evals.eval_scripts.components.eval_generator", language="bash")
        st.caption(f"Looking for timestamped reports under `{RESULTS_DIR.relative_to(BASE_DIR)}/<suite>/`.")
        return

    st.subheader("Latest run by evaluation suite")
    overview = []
    for suite_name, runs in suites.items():
        path, report = runs[-1]
        passed = int(report.get("passed", 0))
        total = int(report.get("total", len(report.get("results", []))))
        overview.append(
            {
                "Evaluation": suite_name.replace("_", " ").title(),
                "Pass rate": _rate(report),
                "Passed": f"{passed}/{total}",
                "Latest run": path.stem,
                "Judge model": report.get("model", "unknown"),
            }
        )
    st.dataframe(
        pd.DataFrame(overview).set_index("Evaluation"),
        use_container_width=True,
        column_config={"Pass rate": st.column_config.ProgressColumn("Pass rate", min_value=0, max_value=100, format="%.1f%%")},
    )

    selected_suite = st.selectbox(
        "Evaluation suite",
        options=list(suites),
        format_func=lambda name: name.replace("_", " ").title(),
    )
    runs = suites[selected_suite]
    history = [
        {
            "Run": path.stem,
            "Pass rate (%)": _rate(report),
            "Passed": int(report.get("passed", 0)),
            "Total": int(report.get("total", len(report.get("results", [])))),
            "Chunk size": report.get("hyperparameters", {}).get("retrieval", {}).get("chunk_size", "Not recorded"),
            "Chunk overlap": report.get("hyperparameters", {}).get("retrieval", {}).get("chunk_overlap", "Not recorded"),
            "Top k": report.get("hyperparameters", {}).get("retrieval", {}).get("top_k", "Not recorded"),
        }
        for path, report in runs
    ]
    st.subheader("Run history and retrieval settings")
    st.dataframe(pd.DataFrame(history).set_index("Run"), use_container_width=True)

    selected_path = st.selectbox(
        "Run details",
        options=[path for path, _ in reversed(runs)],
        format_func=lambda path: path.stem,
    )
    selected_report = next(report for path, report in runs if path == selected_path)
    results = selected_report.get("results", [])
    passed = int(selected_report.get("passed", 0))
    total = int(selected_report.get("total", len(results)))
    first_run, second_run, third_run = st.columns(3)
    first_run.metric("Pass rate", f"{_rate(selected_report):.1f}%")
    second_run.metric("Passed cases", f"{passed} / {total}")
    third_run.metric("Judge model", selected_report.get("model", "unknown"))

    st.subheader("Hyperparameters for this run")
    hyperparameters = selected_report.get("hyperparameters")
    if hyperparameters:
        retrieval = hyperparameters.get("retrieval", {})
        retrieval_columns = st.columns(3)
        retrieval_columns[0].metric("Chunk size", retrieval.get("chunk_size", "Not recorded"))
        retrieval_columns[1].metric("Chunk overlap", retrieval.get("chunk_overlap", "Not recorded"))
        retrieval_columns[2].metric("Retriever top k", retrieval.get("top_k", "Not recorded"))
        with st.expander("All captured configuration", expanded=False):
            st.json(hyperparameters)
    else:
        st.info("This report predates configuration snapshots, so its run-time hyperparameters were not recorded.")

    failures_only = st.toggle("Show failed cases only", value=False)
    visible_results = [row for row in results if not failures_only or not row.get("pass", False)]
    case_rows = [
        {
            "ID": row.get("id", "unknown"),
            "Result": "PASS" if row.get("pass", False) else "FAIL",
            "Judge reason": row.get("reason", ""),
        }
        for row in visible_results
    ]
    st.subheader("Cases")
    st.dataframe(pd.DataFrame(case_rows), use_container_width=True, hide_index=True)

    for row in visible_results:
        if row.get("pass", False):
            continue
        with st.expander(f"Failed case {row.get('id', 'unknown')} details"):
            st.write(row.get("reason", "No judge rationale was saved."))
            if "response" in row:
                st.markdown("**Assistant response**")
                st.write(row["response"])
            if "retrieved" in row:
                st.markdown("**Retrieved chunks**")
                st.json(row["retrieved"])
