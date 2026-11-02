"""Evaluate retrieval relevance, answer evidence, and expected source coverage."""
import json
from evals.eval_scripts.common import judge, load_cases, report

PROMPT = """You are grading a retrieval result for a company handbook RAG system. Compare query, reference answer, expected source, and retrieved chunks. PASS only if at least one chunk contains direct evidence for the key answer (not merely the same topic) and at least one supporting chunk has a source/page matching the expected source where metadata is present. Do not require exact wording; numbers, names, and qualifiers must be present when material. Missing source metadata alone is not a failure if the chunk itself clearly identifies the expected page/source. Return only JSON: {\"pass\": boolean, \"reason\": \"one concise sentence\"}.\nCase: {case}\nRetrieved: {retrieved}"""

def main():
    from src.services.index_docs import get_retriever
    retriever = get_retriever()
    if retriever is None:
        raise RuntimeError("Qdrant collection unavailable; start Qdrant and index the handbook first.")
    rows = []
    for case in load_cases("retriever_golden.json"):
        docs = retriever.invoke(case["query"])
        found = [{"text": d.page_content, "metadata": d.metadata} for d in docs]
        prompt = PROMPT.replace("{case}", json.dumps(case, ensure_ascii=False)).replace(
            "{retrieved}", json.dumps(found, ensure_ascii=False)
        )
        result = judge(prompt)
        rows.append({"id": case["id"], "pass": result["pass"], "reason": result.get("reason", ""), "retrieved": found})
    return report("retriever", rows)

if __name__ == "__main__": raise SystemExit(main())
