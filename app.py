import os
import time
import uuid

import psycopg
import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from database import (
    DB_URI,
    add_message,
    create_thread,
    get_all_threads,
    get_messages,
    init_db,
    update_thread_title,
)
from graph_builder import build_graph
from index_docs import index_all_documents
from redis_store import RedisSemanticCache, RedisSessionManager

st.set_page_config(page_title="Company On-boarding RAG", layout="wide")

load_dotenv()


@st.cache_resource
def setup_environment():
    init_db()

    with psycopg.connect(DB_URI, autocommit=True) as conn:
        PostgresSaver(conn).setup()

    semantic_cache_threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.88"))
    session_manager = RedisSessionManager()
    semantic_cache = RedisSemanticCache(threshold=semantic_cache_threshold)
    checkpoint_pool = ConnectionPool(conninfo=DB_URI, max_size=20)
    checkpointer = PostgresSaver(checkpoint_pool)
    graph = build_graph(checkpointer, session_manager, semantic_cache)
    return graph, session_manager, semantic_cache


app, session_manager, semantic_cache = setup_environment()

if "current_thread_id" not in st.session_state:
    threads = get_all_threads()
    all_thread_ids = {thread_id for thread_id, _ in threads}
    url_thread_id = st.query_params.get("thread_id")
    if url_thread_id and url_thread_id in all_thread_ids:
        st.session_state.current_thread_id = url_thread_id
    else:
        st.session_state.current_thread_id = str(uuid.uuid4())
        create_thread(st.session_state.current_thread_id, "New Chat")

st.query_params["thread_id"] = st.session_state.current_thread_id

with st.sidebar:
    st.header("Control Panel")

    if st.button("Index Documents Here", use_container_width=True):
        with st.spinner("Extracting and indexing company documents..."):
            chunks_indexed = index_all_documents()
            if chunks_indexed > 0:
                st.success(f"Successfully indexed {chunks_indexed} chunks.")
            else:
                st.warning("No supported company documents found in ./documents.")

    st.divider()

    st.subheader("Chat History")
    if st.button("New Chat", use_container_width=True):
        st.session_state.current_thread_id = str(uuid.uuid4())
        create_thread(st.session_state.current_thread_id, "New Chat")
        st.query_params["thread_id"] = st.session_state.current_thread_id
        st.rerun()

    threads = get_all_threads()
    for thread_id, title in threads:
        btn_label = title if title and title != "New Chat" else f"Chat {thread_id[:5]}"
        if st.button(btn_label, key=thread_id, use_container_width=True):
            st.session_state.current_thread_id = thread_id
            st.query_params["thread_id"] = thread_id
            st.rerun()

st.title("Company On-boarding RAG")
st.caption(f"Current Session: `{st.session_state.current_thread_id}`")

messages = get_messages(st.session_state.current_thread_id)
for role, content in messages:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Ask a question..."):
    add_message(st.session_state.current_thread_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    initial_state = {
        "question": prompt,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
        "cached_answer": "",
        "cache_score": 0.0,
        "response_source": "",
    }
    config = {
        "configurable": {
            "thread_id": st.session_state.current_thread_id,
            "user_id": "12345",
        },
        "metadata": {"thread_id": st.session_state.current_thread_id},
        "run_name": "chat_turn",
    }

    final_answer = ""
    cache_score = 0.0
    response_source = ""

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            for event in app.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, state_update in event.items():
                    st.write(f"Completed step: `{node_name}`")
                    if state_update.get("cache_score"):
                        cache_score = float(state_update["cache_score"])
                    if state_update.get("response_source"):
                        response_source = state_update["response_source"]
                    if state_update.get("answer"):
                        final_answer = state_update["answer"]
            status.update(label="Complete", state="complete", expanded=False)

        if response_source == "semantic_cache" and cache_score:
            st.caption(f"Redis semantic cache hit (similarity: {cache_score:.3f})")

        def _stream_answer(text: str):
            for line in text.splitlines(keepends=True):
                for word in line.split(" "):
                    yield word + " "
                    time.sleep(0.02)

        st.write_stream(_stream_answer(final_answer))

    add_message(st.session_state.current_thread_id, "assistant", final_answer)

    if len(messages) == 0:
        short_title = prompt[:25] + "..." if len(prompt) > 25 else prompt
        update_thread_title(st.session_state.current_thread_id, short_title)
        st.rerun()
