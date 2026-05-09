"""
SG-ComplianceGuard — Streamlit Frontend
"""

import os
from urllib.parse import quote
import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MAX_HISTORY_MESSAGES = 1000

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document RAG",
    page_icon="📄",
    layout="wide",
)

# Keep Streamlit settings menu (theme switcher) but hide Deploy button.
st.markdown(
    """
    <style>
    [data-testid="stAppDeployButton"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 LOCAL RAG")
st.caption(
    "Upload your documents (PDF, CSV, Excel, TXT) and ask questions about them. "
    "All processing and indexing is done locally."
)
st.divider()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Sidebar — File Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "csv", "xlsx", "xls", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Index", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        resp = httpx.post(f"{BACKEND_URL}/upload", files=files, timeout=300.0)
                        resp.raise_for_status()
                        st.success(f"Indexed {uploaded_file.name} ({resp.json()['chunks']} chunks)")
                    except Exception as e:
                        st.error(f"Error indexing {uploaded_file.name}: {e}")

    st.divider()
    st.header("🗂 Indexed Files")
    try:
        docs_resp = httpx.get(f"{BACKEND_URL}/documents", timeout=20.0)
        docs_resp.raise_for_status()
        indexed_docs = docs_resp.json()
    except Exception as e:
        indexed_docs = []
        st.caption(f"Could not load indexed files: {e}")

    if not indexed_docs:
        st.caption("No files indexed yet.")
    else:
        for doc in indexed_docs:
            col_name, col_action = st.columns([4, 1])
            with col_name:
                st.caption(f"`{doc['filename']}` ({doc['chunks']} chunks)")
            with col_action:
                if st.button("🗑", key=f"del_{doc['filename']}", help="Delete file from vector store"):
                    try:
                        encoded = quote(doc["filename"], safe="")
                        del_resp = httpx.delete(f"{BACKEND_URL}/documents/{encoded}", timeout=20.0)
                        del_resp.raise_for_status()
                        st.success(f"Deleted {doc['filename']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- 🔍 Qdrant (vector DB)\n"
        "- 🤗 all-MiniLM-L6-v2 (embeddings)\n"
        "- 🦙 local LLM via Ollama\n"
        "- ⚡ FastAPI backend\n"
        "- 🔒 100% local — PDPA compliant"
    )
    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.header("💬 Conversation History")
    if not st.session_state.history:
        st.caption("No messages yet.")
    else:
        for idx, entry in enumerate(st.session_state.history[-12:], start=max(1, len(st.session_state.history) - 11)):
            preview = entry["question"].strip().replace("\n", " ")
            if len(preview) > 55:
                preview = preview[:55] + "..."
            st.caption(f"{idx}. {preview}")

# ---------------------------------------------------------------------------
# Chat transcript
# ---------------------------------------------------------------------------
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])

    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry["sources"]:
            with st.expander("Sources", expanded=False):
                st.caption("Context used for this answer:")
                st.caption("`Rank score` is retrieval ordering from hybrid search, not answer certainty.")
                for i, src in enumerate(entry["sources"]):
                    st.markdown(f"**Source {i + 1}:** `{src['source_url']}`")
                    st.caption(f"Rank score: {src['score']:.4f}")
                    st.markdown(src["text"])
                    if i < len(entry["sources"]) - 1:
                        st.markdown("---")

if not st.session_state.history:
    st.info("Upload documents in the sidebar and ask a question above to get started.")

# ---------------------------------------------------------------------------
# Chat input + query execution
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question about your documents...")

if question and question.strip():
    history_payload = []
    for turn in st.session_state.history:
        history_payload.append({"role": "user", "content": turn["question"]})
        history_payload.append({"role": "assistant", "content": turn["answer"]})

    if len(history_payload) >= MAX_HISTORY_MESSAGES:
        st.error(
            f"Message too long. This conversation has reached {MAX_HISTORY_MESSAGES} messages. "
            "Please clear chat history to continue."
        )
        st.stop()

    with st.spinner("Searching documents and generating answer..."):
        try:
            resp = httpx.post(
                f"{BACKEND_URL}/query",
                json={"question": question.strip(), "history": history_payload},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.history.append({
                "question": question.strip(),
                "answer": data["answer"],
                "sources": data["sources"],
            })
            st.rerun()
        except httpx.ConnectError:
            st.error("Cannot connect to the backend. Make sure the FastAPI server is running.")
        except Exception as e:
            st.error(f"Error: {e}")
