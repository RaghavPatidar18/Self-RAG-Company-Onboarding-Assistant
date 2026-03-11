# Fresher Training RAG: Self-RAG Company Onboarding Assistant

A production-grade **Self-RAG** chatbot for fresher onboarding and internal knowledge support.  
It combines retrieval, self-evaluation, short-term memory, long-term memory, and semantic caching to improve answer quality, personalization, and latency.

---

## Latest Architecture

The app is a **LangGraph state machine** with three memory layers:
- **Short-term memory:** per-thread rolling summary in graph state (`summary`)
- **Long-term memory:** persistent user facts in `PostgresStore`
- **Semantic cache:** vector-similarity answer cache for repeated/similar prompts

```text
User Prompt
   |
   +--> [Semantic Cache Lookup (Qdrant)] -- hit --> return cached answer
   |                                          
   |-- miss --> LangGraph Pipeline ------------------------------------------+
            [decide_retrieval]                                               |
              |                                                               |
              +--> [generate_direct]                                          |
              |         |                                                     |
              |         v                                                     |
              |    [gen_summary]  (short-term memory update)                  |
              |         |                                                     |
              |         v                                                     |
              |    [gen_ltm]      (long-term memory write)                    |
              |         |                                                     |
              |        END                                                    |
              |
              +--> [retrieve (Qdrant KB)] -> [is_relevant] -> [generate_from_context]
                                                      |                |
                                                      |                +-- uses:
                                                      |                    - relevant docs
                                                      |                    - short-term summary
                                                      |                    - long-term user memory
                                                      v
                                                   [is_sup] -> [revise_answer] (max 3)
                                                      |
                                                   [is_use]
                                                      |
                                     +----------------+----------------+
                                     |                                 |
                                   useful                        not_useful
                                     |                                 |
                               [gen_summary]                     [rewrite_question]
                                     |                                 |
                               [gen_ltm] <--------------------- [retrieve retry]
                                     |
                                    END

After successful generation:
- answer is written to semantic cache
- user/assistant messages are stored in Postgres chat tables
```

---

## Component Overview

| Component | File | Responsibility |
|---|---|---|
| Streamlit UI | `app.py` | Chat UI, thread/session handling, graph streaming |
| Graph Orchestration | `graph_builder.py` | Self-RAG nodes, routing logic, memory read/write |
| Semantic Cache | `semantic_cache.py` | Similarity cache using embeddings + Qdrant |
| Document Indexer/Retriever | `index_docs.py` | OCR text chunking, embeddings, Qdrant upsert/query |
| OCR Reader | `document_reader.py` | PDF/PPTX/TXT extraction (EasyOCR + parsers) |
| Chat Persistence | `database.py` | `chat_threads` / `chat_messages` Postgres storage |

---

## Memory and Cache Layers

### 1. Short-Term Memory
- Implemented as `summary` in graph state.
- `gen_summary` creates a concise summary from the latest Q/A.
- Next turns can use this summary to keep context continuity inside the active thread.

### 2. Long-Term Memory
- Implemented in `gen_ltm` via `PostgresStore`.
- Extracts durable user facts (identity, preferences, ongoing goals).
- Stores atomic memory entries under a user namespace and reuses them in `generate_from_context` for personalization.

### 3. Semantic Cache
- Implemented in `semantic_cache.py`.
- Embeds incoming question and checks nearest cached question by cosine similarity.
- Returns cached answer if score is above threshold (`SEMANTIC_CACHE_THRESHOLD`, default `0.88`).
- On cache miss, normal graph executes; successful answers are then cached.

### 4. Qdrant DB ("Quadrant DB" -> Qdrant)
- Qdrant is used in two places:
  - **Knowledge base retrieval** (`index_docs.py`) for document chunks.
  - **Semantic cache storage** (`semantic_cache.py`) for Q/A reuse.

---

## Problems Solved by These Changes

### 1. Repeated Questions Causing Unnecessary LLM Calls
**Problem:** Similar questions repeatedly triggered full retrieval + generation, increasing latency and cost.  
**Solved by:** Semantic cache layer that serves high-similarity cached answers.

### 2. Context Loss Across Turns
**Problem:** Multi-turn responses could become inconsistent without a compact running context.  
**Solved by:** Short-term summary memory (`gen_summary`) reused in later steps.

### 3. No Durable Personalization
**Problem:** User-specific details were not retained as reusable durable facts.  
**Solved by:** Long-term memory extraction and persistence in `PostgresStore` (`gen_ltm`).

### 4. Generic Responses Despite Known User Context
**Problem:** Even with prior interactions, responses remained generic.  
**Solved by:** `generate_from_context` now includes long-term memory + short-term summary with retrieved docs.

### 5. Retrieval Misses From Poorly Worded Queries
**Problem:** Conversational prompts often underperform in vector search.  
**Solved by:** `rewrite_question` retries with optimized retrieval query terms.

### 6. Hallucinated or Weakly Grounded Answers
**Problem:** Generated responses could include unsupported claims.  
**Solved by:** `is_sup` grounding check + `revise_answer` correction loop (up to 3 retries).

### 7. Helpful-but-Not-Actually-Useful Answers
**Problem:** Factually grounded answers could still fail user intent.  
**Solved by:** Separate usefulness gate (`is_use`) before final acceptance.

---

## Tech Stack

| Category | Technology |
|---|---|
| LLM | Groq API — `openai/gpt-oss-120b` |
| Orchestration | LangGraph (StateGraph) |
| LLM Framework | LangChain |
| Embeddings | HuggingFace — `all-MiniLM-L6-v2` |
| Vector DB | Qdrant |
| OCR | EasyOCR |
| UI | Streamlit |
| Persistence | PostgreSQL (`psycopg`, `PostgresSaver`, `PostgresStore`) |
| Containerization | Docker Compose |

---

## Setup (Local)

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- Poppler (for `pdf2image`)
- Groq API key

### 1. Start infrastructure

```bash
docker-compose up -d
```

Starts:
- PostgreSQL on `5432`
- Qdrant on `6333`/`6334`

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure `.env`

```env
GROQ_API_KEY=your_groq_api_key_here

# Retrieval Qdrant
QDRANT_COLLECTION_NAME=ocr_chunks
QDRANT_URL=http://localhost:6333

# Postgres
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_db
DB_PORT=5432

# Semantic cache
SEMANTIC_CACHE_THRESHOLD=0.88
SEMANTIC_CACHE_COLLECTION=semantic_cache
# Optional local persistence for cache; default is in-memory
# SEMANTIC_CACHE_PATH=./cache_qdrant
```

### 4. Add documents
Put PDF/TXT/PPTX files in `./documents/`.

### 5. Run app

```bash
streamlit run app.py
```

### 6. Build retrieval index
In sidebar, click **Index Documents Here**.

---

## Runtime Flow Summary

1. User asks question.
2. Semantic cache checks for similar prior question.
3. On miss, LangGraph runs retrieval/direct-answer path.
4. Grounding and usefulness checks enforce quality.
5. Summary and long-term memory are updated.
6. Final answer and messages are persisted.
7. Answer is added to semantic cache.

