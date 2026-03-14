"""
SG-ComplianceGuard — Streamlit Frontend
"""

import os
import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document RAG",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 Document RAG Assistant")
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
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

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
    st.markdown("**Stack**")
    st.markdown(
        "- 🔍 Qdrant (vector DB)\n"
        "- 🤗 all-MiniLM-L6-v2 (embeddings)\n"
        "- 🦙 local LLM via Ollama\n"
        "- ⚡ FastAPI backend\n"
        "- 🔒 100% local — PDPA compliant"
    )

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
question = st.text_input(
    "Ask a question about your documents:",
    value=st.session_state.current_question,
    placeholder="e.g. What is the summary of the financial report?",
    key="question_input",
)

ask_col, clear_col = st.columns([1, 5])
with ask_col:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with clear_col:
    if st.button("Clear history", use_container_width=False):
        st.session_state.history = []
        st.session_state.current_question = ""
        st.rerun()

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
if ask_clicked and question.strip():
    st.session_state.current_question = question.strip()
    with st.spinner("Searching documents and generating answer…"):
        try:
            resp = httpx.post(
                f"{BACKEND_URL}/query",
                json={"question": question.strip()},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            st.session_state.history.insert(0, {
                "question": question.strip(),
                "answer": data["answer"],
                "sources": data["sources"],
            })
        except httpx.ConnectError:
            st.error("Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
for entry in st.session_state.history:
    st.subheader(f"Q: {entry['question']}")

    answer_col, source_col = st.columns([1, 1], gap="large")

    with answer_col:
        st.markdown("#### 🤖 AI Answer")
        st.markdown(entry["answer"])

    with source_col:
        st.markdown("#### 📄 Source Verification")
        st.caption("Context from uploaded documents:")

        for i, src in enumerate(entry["sources"]):
            with st.expander(
                f"Source {i + 1} — {src['source_url']}",
                expanded=(i == 0),
            ):
                st.caption(f"Relevance score: {src['score']:.4f}")
                st.markdown("---")
                st.markdown(src["text"])

    st.divider()

if not st.session_state.history:
    st.info("Upload documents in the sidebar and ask a question above to get started.")
