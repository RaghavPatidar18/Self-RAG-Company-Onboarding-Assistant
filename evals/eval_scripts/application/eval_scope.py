"""Evaluate company-assistant scope behavior against scope_golden.json."""
import json
from evals.eval_scripts.common import app_answer, judge, load_cases, report

PROMPT = """Act as a strict evaluator for an internal company onboarding assistant. Compare request, expected_action, success_criteria, and response. Decide pass only when all criteria match: ANSWER means substantive company-related help; DECLINE means no substantive execution of out-of-scope request; PARTIAL means the in-scope part is answered and out-of-scope part is not fulfilled. A polite boundary statement is fine. Incidental mention is not fulfillment. Use no assumptions beyond the supplied fields. Return only JSON: {\"pass\": boolean, \"reason\": \"one concise sentence\"}.\nCase: {case}\nResponse: {response}"""

def main():
    rows = []
    for case in load_cases("scope_golden.json"):
        answer = app_answer(case["input"], f"eval-scope-{case['id']}")
        prompt = PROMPT.replace("{case}", json.dumps(case, ensure_ascii=False)).replace("{response}", answer)
        result = judge(prompt)
        rows.append({"id": case["id"], "pass": result["pass"], "reason": result.get("reason", ""), "response": answer})
    return report("scope", rows)

if __name__ == "__main__": raise SystemExit(main())
