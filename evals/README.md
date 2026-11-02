# LLM-as-judge evaluations

The scripts under `eval_scripts/application` evaluate assistant scope, toxicity, and disclosure behavior. The component scripts evaluate handbook retrieval and answer generation. Application and generator suites invoke the production LangGraph; retrieval invokes the production Qdrant retriever. Each suite uses the corresponding JSON golden file in `eval_dataset` and writes per-case responses, judge reasons, and pass/fail results to a timestamped file in `evals/results/<suite>/`. Each execution gets a unique filename, so prior runs remain intact.

## Judge design and acceptance

Every judge prompt is task-specific and includes explicit pass/fail rules derived from the golden case's expected action or reference answer. The judge must emit a JSON object with a boolean `pass` and concise `reason`; malformed output stops the run instead of being silently counted. Evaluation calls use Groq with temperature 0 and a pinned default model (`openai/gpt-oss-120b`), configurable using `EVAL_GROQ_MODEL`. A suite passes only when every golden case passes. Temperature zero and fixed model reduce rerun variance, but hosted model updates can still change judgments; retain the result JSON and model name for comparisons.

## Run

1. Install the repository dependencies in the project's Python environment (`pip install -r requirements.txt`).
2. Set `GROQ_API_KEY` in the project `.env` file or shell. Optionally set `EVAL_GROQ_MODEL` to a Groq-supported model identifier.
3. Start the services required by the app: PostgreSQL, Redis, and Qdrant. Ensure the company handbook is present under `documents/` and indexed in Qdrant before running retrieval/generator/application suites.
4. From the repository root, run one suite at a time:

   ```bash
   python -m evals.eval_scripts.application.eval_scope
   python -m evals.eval_scripts.application.eval_toxicity
   python -m evals.eval_scripts.application.eval_leakage
   python -m evals.eval_scripts.components.eval_retriever
   python -m evals.eval_scripts.components.eval_generator
   ```

5. Check the process exit code (0 means all cases passed; 1 means at least one case failed) and inspect the corresponding folder under `evals/results/` for response and judge rationale details. Files are named like `scope_2026-11-02_15-45-30_IST.json`.

Each run makes live Groq calls. Application suites also rely on configured database/cache services and may use model embeddings for cache initialization.
