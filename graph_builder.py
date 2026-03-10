import os
from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from langsmith import traceable

from psycopg_pool import ConnectionPool
from index_docs import get_retriever
from database import DB_URI

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

class State(TypedDict):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: str
    evidence: List[str]
    retries: int
    isuse: str
    use_reason: str

# --- 1. Nodes & Edges (Adapted from your code) ---
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(..., description="True if external documents are needed.")

@traceable(name="decide_retrival")
def decide_retrieval(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI routing agent for an agricultural business. 
        Determine if the user's question requires retrieving specialized documents (e.g., organic farming guidelines, localized climate data, crop yield reports, or community agribusiness initiatives). 
        Return JSON with 'should_retrieve' (boolean). 
        Set to True if it needs specific, localized, or proprietary database info. 
        Set to False for general greetings or broad, universally known concepts."""),
        ("human", "Question: {question}")
    ])
    decision = llm.with_structured_output(RetrieveDecision).invoke(prompt.format_messages(question=state["question"]))
    return {"need_retrieval": decision.should_retrieve}

@traceable(name="route decision")
def route_after_decide(state: State):
    return "retrieve" if state["need_retrieval"] else "generate_direct"

@traceable(name="generate with llm knowledge")
def generate_direct(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an agricultural expert assistant. 
        Answer the user's question using your general knowledge about farming, crops, livestock, and agribusiness. 
        If the question requires highly localized data, specific chemical regulations, or proprietary agribusiness information that you do not possess, politely state 'I don't know' or 'I need more specific agricultural context to answer this'."""),
        ("human", "{question}")
    ])
    return {"answer": llm.invoke(prompt.format_messages(question=state["question"])).content}

@traceable(name="retrieve doc")
def retrieve(state: State):
    retriever = get_retriever()
    if not retriever:
        return {"docs": []}
    q = state.get("retrieval_query") or state["question"]
    docs = retriever.invoke(q)
    return {"docs": docs}

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(..., description="True ONLY if document directly relates to the question topic.")

@traceable(name="is relevant")
def is_relevant(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an agricultural researcher evaluating document relevance. 
        Assess if the provided document contains information directly relevant to answering the user's farming or agribusiness question. 
        Return JSON with 'is_relevant' (boolean). 
        Set to True ONLY if the document discusses the specific crops, techniques (like organic composting), equipment, or agricultural concepts mentioned in the question. False if it is off-topic."""),
        ("human", "Question:\n{question}\n\nDoc:\n{document}")
    ])
    relevant_docs = []
    for doc in state.get("docs", []):
        decision = llm.with_structured_output(RelevanceDecision).invoke(
            prompt.format_messages(question=state["question"], document=doc.page_content)
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}

@traceable(name="rout after relevance")
def route_after_relevance(state: State):
    return "generate_from_context" if state.get("relevant_docs") else "no_answer_found"

@traceable(name="generate_from_context")
def generate_from_context(state: State):
    context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialized agricultural AI assistant. 
        Answer the user's question clearly and accurately using ONLY the provided agricultural context. Focus on actionable insights for farming or agribusiness if present. 
        Do not use outside knowledge. If the context does not contain the answer, state that clearly. 
        Do not use conversational filler like 'Based on the context provided' or 'According to the documents'."""),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])
    return {"answer": llm.invoke(prompt.format_messages(question=state["question"], context=context)).content, "context": context}

@traceable(name="no answer found")
def no_answer_found(state: State):
    return {"answer": "No answer found in the knowledge base.", "context": ""}

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

@traceable(name="is supporting")
def is_sup(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict fact-checker for an agricultural knowledge base. 
        Verify if the generated ANSWER is strictly supported by the provided CONTEXT. 
        Return JSON with 'issup' (Literal: 'fully_supported', 'partially_supported', 'no_support') and 'evidence' (a list of direct quotes from the context that support the answer). 
        Ensure no unverified farming advice, crop yield guarantees, or chemical recommendations are hallucinated in the ANSWER."""),
        ("human", "Question:\n{question}\nAnswer:\n{answer}\nContext:\n{context}")
    ])
    decision = llm.with_structured_output(IsSUPDecision).invoke(
        prompt.format_messages(question=state["question"], answer=state.get("answer", ""), context=state.get("context", ""))
    )
    return {"issup": decision.issup, "evidence": decision.evidence}

@traceable(name="route after supporting")
def route_after_issup(state: State):
    if state.get("issup") == "fully_supported" or state.get("retries", 0) >= 3:
        return "accept_answer"
    return "revise_answer"

def accept_answer(state: State): return {}

@traceable(name="revise answer")
def revise_answer(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an agricultural AI assistant correcting a previous answer that contained unsupported claims. 
        Rewrite the answer to the question using STRICTLY the facts present in the provided CONTEXT. 
        Remove any hallucinated advice regarding crops, shelf-life, or farming practices. Stay perfectly grounded in the provided text."""),
        ("human", "Question:\n{question}\nAnswer:\n{answer}\nCONTEXT:\n{context}")
    ])
    return {"answer": llm.invoke(prompt.format_messages(question=state["question"], answer=state.get("answer",""), context=state.get("context",""))).content, "retries": state.get("retries", 0) + 1}

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str

@traceable(name="is usefull answer")
def is_use(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are evaluating the helpfulness of an agricultural assistant's response. 
        Decide if the ANSWER effectively and directly addresses the user's QUESTION. 
        Return JSON with 'isuse' (Literal: 'useful', 'not_useful') and 'reason' (a brief explanation). 
        Consider it 'useful' if it provides actionable, accurate farming or business info relevant to the prompt. Consider it 'not_useful' if it evades the question or provides irrelevant info."""),
        ("human", "Question:\n{question}\nAnswer:\n{answer}")
    ])
    decision = llm.with_structured_output(IsUSEDecision).invoke(prompt.format_messages(question=state["question"], answer=state.get("answer", "")))
    return {"isuse": decision.isuse, "use_reason": decision.reason}

@traceable(name="route after usefulness check")
def route_after_isuse(state: State):
    if state.get("isuse") == "useful": return "END"
    if state.get("rewrite_tries", 0) >= 3: return "no_answer_found"
    return "rewrite_question"

class RewriteDecision(BaseModel):
    retrieval_query: str

@traceable(name="rewrite question")
def rewrite_question(state: State):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at agricultural information retrieval. The previous search query did not yield useful results. 
        Rewrite the original QUESTION into a highly optimized search query for a vector database. 
        Focus on extracting key agricultural terms (e.g., 'organic compost methods', 'high-shelf-life crop processing', 'soil moisture retention'). 
        Remove conversational filler. Return JSON with 'retrieval_query' (string)."""),
        ("human", "Question:\n{question}\nPrev Query:\n{retrieval_query}")
    ])
    decision = llm.with_structured_output(RewriteDecision).invoke(
        prompt.format_messages(question=state["question"], retrieval_query=state.get("retrieval_query", ""))
    )
    return {"retrieval_query": decision.retrieval_query, "rewrite_tries": state.get("rewrite_tries", 0) + 1, "docs": [], "relevant_docs": [], "context": ""}

# --- 2. Build and Compile Graph ---
def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node("decide_retrieval", decide_retrieval)
    g.add_node("generate_direct", generate_direct)
    g.add_node("retrieve", retrieve)
    g.add_node("is_relevant", is_relevant)
    g.add_node("generate_from_context", generate_from_context)
    g.add_node("no_answer_found", no_answer_found)
    g.add_node("is_sup", is_sup)
    # g.add_node("accept_answer", accept_answer)
    g.add_node("revise_answer", revise_answer)
    g.add_node("is_use", is_use)
    g.add_node("rewrite_question", rewrite_question)

    g.add_edge(START, "decide_retrieval")
    g.add_conditional_edges("decide_retrieval", route_after_decide, {"generate_direct": "generate_direct", "retrieve": "retrieve"})
    g.add_edge("generate_direct", END)
    g.add_edge("retrieve", "is_relevant")
    g.add_conditional_edges("is_relevant", route_after_relevance, {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"})
    g.add_edge("no_answer_found", END)
    g.add_edge("generate_from_context", "is_sup")
    g.add_conditional_edges("is_sup", route_after_issup, {"accept_answer": "is_use", "revise_answer": "revise_answer"})
    g.add_edge("revise_answer", "is_sup")
    g.add_conditional_edges("is_use", route_after_isuse, {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"})
    g.add_edge("rewrite_question", "retrieve")

    return g.compile(checkpointer=checkpointer)

# Manage Postgres connection pool for the checkpointer globally
pool = ConnectionPool(conninfo=DB_URI, max_size=20)