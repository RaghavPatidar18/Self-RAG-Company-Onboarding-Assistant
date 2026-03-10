import streamlit as st
import uuid
import time
from database import init_db, create_thread, get_all_threads, add_message, get_messages
from graph_builder import build_graph, pool
from index_docs import index_all_documents
from semantic_cache import SemanticCache
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from database import DB_URI

st.set_page_config(page_title="Agri-RAG", layout="wide")

import os
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize DB and Graph Checkpointer
@st.cache_resource # runs only once on server up
def setup_environment():
    init_db()
    
    # 1. Run the LangGraph table migrations using a dedicated autocommit connection
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        setup_checkpointer = PostgresSaver(conn)
        setup_checkpointer.setup() 
        
    # 2. Instantiate the actual checkpointer for the graph using the connection pool
    checkpointer = PostgresSaver(pool)
    semantic_cache_threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.88"))
    semantic_cache = SemanticCache(threshold=semantic_cache_threshold)
    return build_graph(checkpointer), semantic_cache

app, semantic_cache = setup_environment()

# 2. Session State Management — restore from URL so reloads don't create new threads
if "current_thread_id" not in st.session_state:
    threads = get_all_threads()
    all_thread_ids = {t[0] for t in threads}
    url_thread_id = st.query_params.get("thread_id")
    if url_thread_id and url_thread_id in all_thread_ids:
        st.session_state.current_thread_id = url_thread_id
    else:
        st.session_state.current_thread_id = str(uuid.uuid4())
        create_thread(st.session_state.current_thread_id, "New Chat")

# Keep the URL in sync so reloads restore the correct thread
st.query_params["thread_id"] = st.session_state.current_thread_id

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Index Documents Button
    if st.button("📥 Index Documents Here", use_container_width=True):
        with st.spinner("Extracting & Embedding PDFs..."):
            chunks_indexed = index_all_documents()
            if chunks_indexed > 0:
                st.success(f"Successfully indexed {chunks_indexed} chunks!")
            else:
                st.warning("No PDFs found in ./documents.")

    st.divider()
    
    # Thread Management
    st.subheader("💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_thread_id = str(uuid.uuid4())
        create_thread(st.session_state.current_thread_id, "New Chat")
        st.query_params["thread_id"] = st.session_state.current_thread_id
        st.rerun()

    threads = get_all_threads()
    for thread_id, title in threads:
        btn_label = f"🗨️ {title}" if title != "New Chat" else f"🗨️ Chat {thread_id[:5]}"
        if st.button(btn_label, key=thread_id, use_container_width=True):
            st.session_state.current_thread_id = thread_id
            st.query_params["thread_id"] = thread_id
            st.rerun()

# --- MAIN CHAT AREA ---
st.title("Agri-RAG")
st.caption(f"Current Session: `{st.session_state.current_thread_id}`")

# Display historical messages from Postgres
messages = get_messages(st.session_state.current_thread_id)
for role, content in messages:
    with st.chat_message(role):
        st.markdown(content)

# Handle new user input
if prompt := st.chat_input("Ask a question..."):
    # Add User Message
    add_message(st.session_state.current_thread_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initial State for Graph Execution
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
    }
    config = {
        "configurable": {"thread_id": st.session_state.current_thread_id},
        "metadata": {
            "thread_id": st.session_state.current_thread_id
        },
        "run_name": "chat_turn",
        }

    final_answer = ""
    cache_hit = None
    try:
        cache_hit = semantic_cache.get(prompt)
    except Exception:
        cache_hit = None

    # Cache-first response path
    if cache_hit:
        cached_answer, cached_score = cache_hit
        final_answer = cached_answer
        with st.chat_message("assistant"):
            st.caption(f"Semantic cache hit (score: {cached_score:.3f})")
            st.markdown(final_answer)
    else:
        # Stream the Graph Response
        with st.chat_message("assistant"):
            # We use st.status to neatly tuck away LangGraph's processing states
            with st.status("Thinking...", expanded=True) as status:
                # Stream mode 'updates' allows us to peek into the nodes being executed
                for event in app.stream(initial_state, config=config, stream_mode="updates"):
                    for node_name, state_update in event.items():
                        st.write(f"✓ Completed step: `{node_name}`")
                        
                        # Capture the latest answer generated
                        if "answer" in state_update and state_update["answer"]:
                            final_answer = state_update["answer"]
                
                status.update(label="Complete!", state="complete", expanded=False)

            # Stream the final answer while preserving markdown structure
            def _stream_answer(text):
                for line in text.splitlines(keepends=True):
                    for word in line.split(" "):
                        yield word + " "
                        time.sleep(0.02)

            st.write_stream(_stream_answer(final_answer))

        # Update semantic cache with the latest successful graph answer
        try:
            semantic_cache.add(prompt, final_answer)
        except Exception:
            pass
        
    # Save Assistant Message to Database
    add_message(st.session_state.current_thread_id, "assistant", final_answer)
    
    # If it's a new chat, update title based on first prompt
    if len(messages) == 0:
        short_title = prompt[:25] + "..." if len(prompt) > 25 else prompt
        with psycopg.connect(DB_URI) as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE chat_threads SET title = %s WHERE thread_id = %s", 
                            (short_title, st.session_state.current_thread_id))
            conn.commit()
        st.rerun()
