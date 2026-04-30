import os
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field

from database import (
    get_long_term_memories,
    get_thread_summary,
    upsert_long_term_memories,
    upsert_thread_summary,
)
from index_docs import get_retriever
from redis_store import RedisSemanticCache, RedisSessionManager

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

MAX_SHORT_TERM_TURNS = int(os.getenv("SHORT_TERM_MAX_TURNS", "10"))
RETAINED_SHORT_TERM_TURNS = int(os.getenv("SHORT_TERM_RETAIN_TURNS", "4"))
session_manager_singleton: RedisSessionManager | None = None
semantic_cache_singleton: RedisSemanticCache | None = None


class State(TypedDict, total=False):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: list[Document]
    relevant_docs: list[Document]
    context: str
    answer: str
    issup: str
    evidence: list[str]
    retries: int
    isuse: str
    use_reason: str
    summary: str
    rolling_summary: str
    summarized_turns: int
    short_term_messages: list[dict[str, Any]]
    long_term_memories: list[str]
    memory_context: str
    cached_answer: str
    cache_score: float
    response_source: str


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(..., description="True if external documents are needed.")


class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool = Field(description="True if new, false if duplicate")


class MemoryDecision(BaseModel):
    should_write: bool
    memories: list[MemoryItem] = Field(default_factory=list)


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        ..., description="True ONLY if document directly relates to the question topic."
    )


class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: list[str] = Field(default_factory=list)


class RewriteDecision(BaseModel):
    retrieval_query: str


def _format_transcript(messages: list[dict[str, Any]]) -> str:
    if not messages:
        return "(no recent session turns)"
    return "\n".join(
        f"{message.get('role', 'unknown').upper()}: {message.get('content', '').strip()}"
        for message in messages
        if str(message.get("content", "")).strip()
    )


def _count_turns(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "user")


def _build_memory_context(
    rolling_summary: str, short_term_messages: list[dict[str, Any]], long_term_memories: list[str]
) -> str:
    parts: list[str] = []
    if rolling_summary.strip():
        parts.append(f"Conversation summary:\n{rolling_summary.strip()}")
    if short_term_messages:
        parts.append(f"Recent session turns:\n{_format_transcript(short_term_messages)}")
    if long_term_memories:
        parts.append("Long-term user memory:\n" + "\n".join(f"- {memory}" for memory in long_term_memories))
    return "\n\n".join(parts).strip()


@traceable(name="hydrate_memory")
def hydrate_memory(state: State, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    user_id = config["configurable"]["user_id"]
    session_manager = session_manager_singleton

    try:
        short_term_messages = session_manager.get_messages(thread_id) if session_manager else []
    except Exception:
        short_term_messages = []

    summary_row = get_thread_summary(thread_id)
    rolling_summary = summary_row["summary"] or ""
    summarized_turns = int(summary_row["summarized_turns"] or 0)
    long_term_memories = get_long_term_memories(user_id)

    return {
        "short_term_messages": short_term_messages,
        "rolling_summary": rolling_summary,
        "summarized_turns": summarized_turns,
        "long_term_memories": long_term_memories,
        "memory_context": _build_memory_context(
            rolling_summary=rolling_summary,
            short_term_messages=short_term_messages,
            long_term_memories=long_term_memories,
        ),
    }


@traceable(name="check_semantic_cache")
def check_semantic_cache(state: State, config: RunnableConfig):
    semantic_cache = semantic_cache_singleton
    try:
        hit = semantic_cache.get(state["question"]) if semantic_cache else None
    except Exception:
        hit = None

    if not hit:
        return {"cached_answer": "", "cache_score": 0.0}
    return {"cached_answer": hit.answer, "cache_score": hit.similarity}


def route_after_cache(state: State):
    return "finish_from_cache" if state.get("cached_answer") else "decide_retrieval"


@traceable(name="finish_from_cache")
def finish_from_cache(state: State):
    return {"answer": state["cached_answer"], "response_source": "semantic_cache"}


@traceable(name="decide_retrieval")
def decide_retrieval(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI routing agent for fresher training in a company.
Determine if the user's question requires retrieving specialized company documents.
Use the memory context as supporting context, but set should_retrieve=true whenever the answer depends on internal company material, policies, or proprietary processes.""",
            ),
            ("human", "Question: {question}\n\nMemory Context:\n{memory_context}"),
        ]
    )
    decision = llm.with_structured_output(RetrieveDecision).invoke(
        prompt.format_messages(question=state["question"], memory_context=state.get("memory_context", ""))
    )
    return {"need_retrieval": decision.should_retrieve}


def route_after_decide(state: State):
    return "retrieve" if state["need_retrieval"] else "generate_direct"


@traceable(name="generate_with_llm_knowledge")
def generate_direct(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a fresher training assistant for company onboarding.
Answer the user's question using general professional knowledge plus the supplied user memory.
If the question requires internal company policy, proprietary process, product details, or compliance specifics that are not present, clearly say you need company-specific context.""",
            ),
            ("human", "Question:\n{question}\n\nMemory Context:\n{memory_context}"),
        ]
    )
    response = llm.invoke(
        prompt.format_messages(
            question=state["question"],
            memory_context=state.get("memory_context", ""),
        )
    )
    return {"answer": response.content, "response_source": "graph"}


@traceable(name="retrieve_documents")
def retrieve(state: State):
    retriever = get_retriever()
    if not retriever:
        return {"docs": []}
    query = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(query)}


@traceable(name="filter_relevant_documents")
def is_relevant(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are evaluating company document relevance for fresher training.
Return is_relevant=true only when the document materially helps answer the question.""",
            ),
            ("human", "Question:\n{question}\n\nDoc:\n{document}"),
        ]
    )
    relevant_docs: list[Document] = []
    for doc in state.get("docs", []):
        decision = llm.with_structured_output(RelevanceDecision).invoke(
            prompt.format_messages(question=state["question"], document=doc.page_content)
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


def route_after_relevance(state: State):
    return "generate_from_context" if state.get("relevant_docs") else "no_answer_found"


@traceable(name="no_answer_found")
def no_answer_found(state: State):
    answer = (
        "I could not find a grounded answer in the current knowledge base. "
        "Please add more company-specific material or rephrase the question."
    )
    return {"answer": answer, "context": "", "response_source": "graph"}


@traceable(name="generate_from_context")
def generate_from_context(state: State):
    document_context = "\n\n---\n\n".join(
        doc.page_content for doc in state.get("relevant_docs", [])
    ).strip()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a specialized company training AI assistant.
Answer the user's question clearly and accurately using ONLY the provided company context and user memory.
Use the memory context only for tone, continuity, and personalization. Never let memory override the retrieved company context.
If the retrieved context does not contain the answer, say that clearly.""",
            ),
            (
                "human",
                "Question:\n{question}\n\nMemory Context:\n{memory_context}\n\nCompany Context:\n{context}",
            ),
        ]
    )
    answer = llm.invoke(
        prompt.format_messages(
            question=state["question"],
            memory_context=state.get("memory_context", ""),
            context=document_context,
        )
    ).content
    return {"answer": answer, "context": document_context, "response_source": "graph"}


@traceable(name="verify_support")
def is_sup(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strict fact-checker for a company training knowledge base.
Verify if the answer is strictly supported by the provided context.""",
            ),
            ("human", "Question:\n{question}\nAnswer:\n{answer}\nContext:\n{context}"),
        ]
    )
    decision = llm.with_structured_output(IsSUPDecision).invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}


def route_after_issup(state: State):
    if state.get("issup") == "fully_supported" or state.get("retries", 0) >= 3:
        return "accept_answer"
    return "revise_answer"


@traceable(name="revise_answer")
def revise_answer(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a company training AI assistant correcting unsupported claims.
Rewrite the answer using strictly the facts present in the provided context.""",
            ),
            ("human", "Question:\n{question}\nAnswer:\n{answer}\nContext:\n{context}"),
        ]
    )
    answer = llm.invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    ).content
    return {"answer": answer, "retries": state.get("retries", 0) + 1}


@traceable(name="evaluate_usefulness")
def is_use(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are evaluating whether the answer is directly useful to the user.
Mark it useful only when it addresses the question clearly and actionably.""",
            ),
            ("human", "Question:\n{question}\nAnswer:\n{answer}"),
        ]
    )
    decision = llm.with_structured_output(IsUSEDecision).invoke(
        prompt.format_messages(question=state["question"], answer=state.get("answer", ""))
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}


def route_after_isuse(state: State):
    if state.get("isuse") == "useful":
        return "sync_session_memory"
    if state.get("rewrite_tries", 0) >= 3:
        return "no_answer_found"
    return "rewrite_question"


@traceable(name="rewrite_question")
def rewrite_question(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Rewrite the original question into a retrieval-optimized query for a company knowledge base.
Focus on internal policy, process, and tool keywords.""",
            ),
            (
                "human",
                "Question:\n{question}\nPrevious Query:\n{retrieval_query}\nMemory Context:\n{memory_context}",
            ),
        ]
    )
    decision = llm.with_structured_output(RewriteDecision).invoke(
        prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            memory_context=state.get("memory_context", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


@traceable(name="sync_session_memory")
def sync_session_memory(state: State, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    session_manager = session_manager_singleton

    updated_messages = list(state.get("short_term_messages", [])) + [
        {"role": "user", "content": state["question"]},
        {"role": "assistant", "content": state.get("answer", "")},
    ]

    rolling_summary = state.get("rolling_summary", "")
    summarized_turns = int(state.get("summarized_turns", 0))
    current_turns = _count_turns(updated_messages)

    if current_turns > MAX_SHORT_TERM_TURNS:
        keep_messages = max(RETAINED_SHORT_TERM_TURNS * 2, 2)
        messages_to_keep = updated_messages[-keep_messages:]
        messages_to_summarize = updated_messages[:-keep_messages]

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Maintain a rolling conversation summary for an AI assistant.
Merge the existing summary with the transcript to preserve durable context, open threads, decisions, and user goals.
Keep it concise, factual, and optimized for future prompting.""",
                ),
                (
                    "human",
                    "Existing Summary:\n{existing_summary}\n\nTranscript To Merge:\n{transcript}",
                ),
            ]
        )
        rolling_summary = llm.invoke(
            prompt.format_messages(
                existing_summary=rolling_summary or "(empty)",
                transcript=_format_transcript(messages_to_summarize),
            )
        ).content.strip()
        summarized_turns += _count_turns(messages_to_summarize)
        upsert_thread_summary(thread_id, rolling_summary, summarized_turns)
        updated_messages = messages_to_keep

    if session_manager:
        try:
            session_manager.replace_messages(thread_id, updated_messages)
        except Exception:
            pass

    memory_context = _build_memory_context(
        rolling_summary=rolling_summary,
        short_term_messages=updated_messages,
        long_term_memories=state.get("long_term_memories", []),
    )
    return {
        "short_term_messages": updated_messages,
        "rolling_summary": rolling_summary,
        "summarized_turns": summarized_turns,
        "memory_context": memory_context,
    }


@traceable(name="write_long_term_memory")
def write_long_term_memory(state: State, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]
    existing = "\n".join(state.get("long_term_memories", [])) or "(empty)"

    extractor = llm.with_structured_output(MemoryDecision)
    prompt = """You maintain durable user memory for an AI assistant.

CURRENT LONG-TERM MEMORY:
{existing}

TASK:
- Review the latest user question and assistant answer.
- Extract only stable user facts worth remembering across sessions.
- Focus on identity, stable preferences, ongoing projects, durable goals, and repeated constraints.
- Do not store transient requests, one-off questions, or assistant-generated claims.
- Return atomic sentences.
- Mark is_new=true only when the memory adds meaningfully new information.
- If there is nothing memory-worthy, return should_write=false and an empty list.
"""

    decision: MemoryDecision = extractor.invoke(
        [
            SystemMessage(content=prompt.format(existing=existing)),
            {
                "role": "user",
                "content": f"Latest user question: {state['question']}\nLatest assistant answer: {state.get('answer', '')}",
            },
        ]
    )

    new_memories = [item.text.strip() for item in decision.memories if item.is_new and item.text.strip()]
    if new_memories:
        upsert_long_term_memories(user_id, new_memories, thread_id)
        merged_memories = new_memories + state.get("long_term_memories", [])
    else:
        merged_memories = state.get("long_term_memories", [])

    return {
        "long_term_memories": merged_memories[:25],
        "memory_context": _build_memory_context(
            rolling_summary=state.get("rolling_summary", ""),
            short_term_messages=state.get("short_term_messages", []),
            long_term_memories=merged_memories[:25],
        ),
    }


@traceable(name="update_semantic_cache")
def update_semantic_cache(state: State, config: RunnableConfig):
    if state.get("response_source") != "graph":
        return {"response_source": state.get("response_source")}

    answer = state.get("answer", "").strip()
    if not answer:
        return {"response_source": state.get("response_source")}

    lowered = answer.lower()
    if "could not find a grounded answer" in lowered or "i need company-specific context" in lowered:
        return {"response_source": state.get("response_source")}

    if semantic_cache_singleton:
        try:
            semantic_cache_singleton.add(state["question"], answer, response_source="graph")
        except Exception:
            pass
    return {"response_source": state.get("response_source")}


def build_graph(checkpointer, session_manager: RedisSessionManager, semantic_cache: RedisSemanticCache):
    global session_manager_singleton, semantic_cache_singleton
    session_manager_singleton = session_manager
    semantic_cache_singleton = semantic_cache

    graph = StateGraph(State)
    graph.add_node("hydrate_memory", hydrate_memory)
    graph.add_node("check_semantic_cache", check_semantic_cache)
    graph.add_node("finish_from_cache", finish_from_cache)
    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("generate_direct", generate_direct)
    graph.add_node("retrieve", retrieve)
    graph.add_node("is_relevant", is_relevant)
    graph.add_node("generate_from_context", generate_from_context)
    graph.add_node("no_answer_found", no_answer_found)
    graph.add_node("is_sup", is_sup)
    graph.add_node("revise_answer", revise_answer)
    graph.add_node("is_use", is_use)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("sync_session_memory", sync_session_memory)
    graph.add_node("write_long_term_memory", write_long_term_memory)
    graph.add_node("update_semantic_cache", update_semantic_cache)

    graph.add_edge(START, "hydrate_memory")
    graph.add_edge("hydrate_memory", "check_semantic_cache")
    graph.add_conditional_edges(
        "check_semantic_cache",
        route_after_cache,
        {
            "finish_from_cache": "finish_from_cache",
            "decide_retrieval": "decide_retrieval",
        },
    )
    graph.add_conditional_edges(
        "decide_retrieval",
        route_after_decide,
        {"generate_direct": "generate_direct", "retrieve": "retrieve"},
    )
    graph.add_edge("finish_from_cache", "sync_session_memory")
    graph.add_edge("generate_direct", "sync_session_memory")
    graph.add_edge("retrieve", "is_relevant")
    graph.add_conditional_edges(
        "is_relevant",
        route_after_relevance,
        {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"},
    )
    graph.add_edge("no_answer_found", "sync_session_memory")
    graph.add_edge("generate_from_context", "is_sup")
    graph.add_conditional_edges(
        "is_sup",
        route_after_issup,
        {"accept_answer": "is_use", "revise_answer": "revise_answer"},
    )
    graph.add_edge("revise_answer", "is_sup")
    graph.add_conditional_edges(
        "is_use",
        route_after_isuse,
        {
            "sync_session_memory": "sync_session_memory",
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found",
        },
    )
    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("sync_session_memory", "write_long_term_memory")
    graph.add_edge("write_long_term_memory", "update_semantic_cache")
    graph.add_edge("update_semantic_cache", END)

    return graph.compile(checkpointer=checkpointer)
