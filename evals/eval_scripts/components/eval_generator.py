"""Evaluate answer faithfulness and completeness using the generator golden set."""
import json
from evals.eval_scripts.common import app_answer, evaluate_case, judge, load_cases, report

PROMPT = """Grade a RAG answer against the user question and authoritative ideal_context. PASS only if the response directly answers the question, includes every material fact required by it, and introduces no contradiction or unsupported specific claim. Accept paraphrases and equivalent units/phrasing. If the answer explicitly says it cannot find the fact or declines, fail. Ignore harmless formatting differences. Treat ideal_context as the complete reference; do not add outside knowledge. Return only JSON: {\"pass\": boolean, \"reason\": \"one concise sentence\"}.\nCase: {case}\nResponse: {response}"""

def main():
    rows = []
    for case in load_cases("generator_golden.json"):
        def evaluate(current_case=case):
            answer = app_answer(current_case["query"], f"eval-generator-{current_case['id']}")
            prompt = PROMPT.replace("{case}", json.dumps(current_case, ensure_ascii=False)).replace("{response}", answer)
            result = judge(prompt)
            return {"pass": result["pass"], "reason": result.get("reason", ""), "response": answer}

        rows.append(evaluate_case("generator", case["id"], evaluate))
    return report("generator", rows)

if __name__ == "__main__": raise SystemExit(main())
