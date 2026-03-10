# Agri-RAG: Self-RAG Agricultural Knowledge Assistant

A production-grade **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** chatbot built for agricultural knowledge management. The system intelligently decides whether to retrieve documents, evaluates their relevance, fact-checks its own answers against source context, and rewrites queries when results are poor — all within a stateful, multi-turn chat interface.

---

## Architecture

The system is built around a **LangGraph state machine** that orchestrates a multi-node Self-RAG pipeline. Each node is a discrete reasoning step, and conditional edges route the flow based on LLM decisions.

```
                         ┌─────────────────────────────────────────────────────────────────┐
                         │                        LANGGRAPH STATE MACHINE                  │
                         │                                                                 │
  User Input             │   ┌─────────────────┐                                          │
──────────────────────►  │   │  decide_retrieval│                                          │
                         │   │  (Router Node)   │                                          │
                         │   └────────┬────────┘                                          │
                         │            │                                                    │
                         │    ┌───────┴────────┐                                          │
                         │    │                │                                           │
                         │    ▼                ▼                                           │
                         │ [retrieve]   [generate_direct]──────────────────────► END      │
                         │    │                                                            │
                         │    ▼                                                            │
                         │ [is_relevant]  ◄── (filter per-doc)                            │
                         │    │                                                            │
                         │    ├── No relevant docs ──► [no_answer_found] ──────► END      │
                         │    │                                                            │
                         │    ▼                                                            │
                         │ [generate_from_context]                                         │
                         │    │                                                            │
                         │    ▼                                                            │
                         │ [is_sup]  ◄──────────── [revise_answer] (up to 3x)            │
                         │    │                          ▲                                 │
                         │    ├── not fully_supported ───┘                                │
                         │    │                                                            │
                         │    ▼                                                            │
                         │ [is_use]                                                        │
                         │    │                                                            │
                         │    ├── useful ──────────────────────────────────────► END      │
                         │    ├── not_useful + retries < 3 ──► [rewrite_question]         │
                         │    │                                       │                   │
                         │    │                                       └──► [retrieve]      │
                         │    └── not_useful + retries >= 3 ──► [no_answer_found]         │
                         └─────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | File | Responsibility |
|---|---|---|
| **Streamlit UI** | `app.py` | Chat interface, thread management, streaming output |
| **Graph Builder** | `graph_builder.py` | LangGraph state machine definition and compilation |
| **Document Indexer** | `index_docs.py` | OCR extraction, chunking, and Qdrant index building |
| **Document Reader** | `document_reader.py` | OCR processing for PDF, PPTX, and TXT files |
| **Database** | `database.py` | PostgreSQL operations for chat threads and messages |

### Data Flow

1. **User asks a question** → Streamlit UI sends it to the LangGraph app
2. **`decide_retrieval`** → LLM decides if specialized docs are needed
3. **If yes → `retrieve`** → Qdrant vector search returns top-k chunks
4. **`is_relevant`** → Each chunk is individually evaluated for relevance
5. **`generate_from_context`** → LLM answers using only relevant chunks
6. **`is_sup`** → LLM fact-checks the answer against source context
7. **`is_use`** → LLM evaluates if the answer actually helps the user
8. **If not useful → `rewrite_question`** → Query is reformulated and retrieval retries
9. Final answer is streamed back and persisted to PostgreSQL

---

## Tech Stack

| Category | Technology |
|---|---|
| **LLM** | Groq API — `openai/gpt-oss-120b` |
| **Orchestration** | LangGraph (StateGraph) |
| **LLM Framework** | LangChain |
| **Embeddings** | HuggingFace — `all-MiniLM-L6-v2` |
| **Vector Store** | Qdrant Cloud (REST API) |
| **OCR Engine** | EasyOCR |
| **PDF Processing** | pdf2image + Poppler |
| **PPTX Processing** | python-pptx |
| **UI** | Streamlit |
| **Database** | PostgreSQL (via psycopg3) |
| **Graph Checkpointing** | LangGraph PostgresSaver |
| **Containerization** | Docker Compose |
| **Env Management** | python-dotenv |

---

## Self-RAG Features

Self-RAG (Self-Reflective RAG) extends standard RAG by making the model **reflect on its own outputs** at each step rather than blindly generating from retrieved context.

### 1. Adaptive Retrieval (`decide_retrieval`)
The system does **not always retrieve**. Before fetching any documents, an LLM call determines whether the question actually requires specialized knowledge or can be answered from general expertise. This avoids wasting latency and avoids injecting irrelevant context into simple questions.

### 2. Per-Document Relevance Filtering (`is_relevant`)
Instead of using all retrieved chunks, every chunk is **individually scored for relevance** against the user's question. Only chunks that directly relate to the topic proceed to generation. This prevents diluting the context with tangentially related information.

### 3. Faithfulness Verification (`is_sup`)
After generating an answer, the system **checks its own faithfulness**. The LLM returns one of:
- `fully_supported` — every claim is grounded in the context
- `partially_supported` — some claims lack evidence
- `no_support` — the answer is not grounded

If not fully supported, the answer is sent to `revise_answer`, which rewrites it using only facts present in the source context (up to 3 revision cycles).

### 4. Utility Evaluation (`is_use`)
Even a factual answer may not be **useful** to the user. A separate LLM call judges whether the answer actually addresses the question. If not useful, instead of giving up, the system escalates to query rewriting.

### 5. Iterative Query Rewriting (`rewrite_question`)
When answers are not useful, the original question is **reformulated into an optimized vector search query** — stripping conversational filler and focusing on agricultural domain keywords. Retrieval then retries with the improved query (up to 3 attempts before falling back to "no answer found").

### 6. Stateful Multi-Turn Conversations
LangGraph's `PostgresSaver` checkpointer persists the full graph state per thread in PostgreSQL. This enables **true conversation continuity** — the graph can resume mid-execution and the chat history survives server restarts.

---

## Steps to Run Locally

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Poppler (required by `pdf2image`)
  - macOS: `brew install poppler`
  - Ubuntu: `sudo apt-get install poppler-utils`
  - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH
- A [Groq API key](https://console.groq.com)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd blog-agent
```

### 2. Start PostgreSQL with Docker

```bash
docker-compose up -d
```

This starts a PostgreSQL 15 instance on port `5432` with the credentials defined in `docker-compose.yml`.

### 3. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR will download its model weights (~100MB) on first run.

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

# Qdrant Cloud (required):
QDRANT_COLLECTION_NAME=ocr_chunks
QDRANT_URL=https://your-cluster-id.us-east.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_cloud_api_key_here
# Optional:
# QDRANT_HTTPS=true
# QDRANT_TIMEOUT=30

POSTGRES_USER="rag_user"
POSTGRES_PASSWORD="rag_password"
POSTGRES_DB="rag_db"
DB_PORT="5432"
```

### 6. Add Your Documents

Place PDF, TXT, or PPTX files into the `./documents/` folder:

```
blog-agent/
└── documents/
    ├── your_report.pdf
    ├── farming_guide.pptx
    └── crop_data.txt
```

### 7. Launch the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 8. Index Your Documents

In the Streamlit sidebar, click **"Index Documents Here"**. The app will:
- Run EasyOCR on every PDF page and PPTX image
- Chunk the extracted text (600 tokens, 150 overlap)
- Embed chunks with `all-MiniLM-L6-v2`
- Save to Qdrant Cloud collection `ocr_chunks`

You are now ready to query your documents.

---

## RAG Improvement Steps

The following deliberate choices were made to improve over a naive RAG baseline:

### Problem 1: Irrelevant Retrieval Degrading Answer Quality
**Standard RAG** always retrieves and always uses what it gets, even if the retrieved chunks are off-topic.

**Fix:** Added `decide_retrieval` to skip retrieval entirely for general questions, and `is_relevant` to filter retrieved chunks individually before generation. Only contextually appropriate chunks reach the LLM.

### Problem 2: Hallucination in Generated Answers
LLMs can "fill in the gaps" with plausible-sounding but unsupported claims, especially in domain-specific topics like agricultural yields and chemical compositions.

**Fix:** The `is_sup` node performs explicit faithfulness checking after generation. If the answer contains unsupported claims, `revise_answer` strips them and regenerates from only the verified source text. This loop runs up to 3 times.

### Problem 3: Factually Correct but Unhelpful Answers
An answer can be grounded in context yet still fail to address the user's actual intent.

**Fix:** The `is_use` node evaluates answer utility separately from faithfulness. This decouples "is it true?" from "does it help?" — two distinct quality dimensions.

### Problem 4: Poor Query Formulation Limiting Recall
User questions phrased conversationally ("what do they say about wheat prices?") do not vector-search well.

**Fix:** The `rewrite_question` node reformulates failed queries into dense, keyword-rich search strings optimized for the Qdrant index. This gives the retriever a second (and third) chance with a better signal.

### Problem 5: Standard Text Extraction Missing Visual Content
PDFs in agricultural reporting often embed charts, tables, and infographics as images. Pure text-based PDF parsing would miss all of this content.

**Fix:** Replaced `PyPDF2`/`pdfplumber` with a full OCR pipeline: PDFs are converted page-by-page to images via `pdf2image`, then processed by `EasyOCR`. PPTX files are similarly handled — native text is extracted from shapes, and embedded images are individually OCR'd.

### Problem 6: Context Lost Across Restarts
In-memory chat state is lost when the server restarts, breaking multi-session continuity.

**Fix:** LangGraph's `PostgresSaver` checkpointer persists the full graph execution state to PostgreSQL per thread. Chat history is also independently stored in `chat_threads` and `chat_messages` tables for the UI layer.
