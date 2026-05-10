"""
SG-ComplianceGuard — Streamlit Frontend
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from urllib.parse import quote
import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MAX_HISTORY_MESSAGES = 1000
CHAT_HISTORY_FILE = Path(__file__).resolve().parent / ".chat_history.json"


def _new_session(title: str = "New chat") -> dict:
    return {
        "id": uuid4().hex[:12],
        "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }


def _normalize_history(raw_history: list) -> list[dict]:
    cleaned: list[dict] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        sources = item.get("sources", [])
        if isinstance(question, str) and isinstance(answer, str) and isinstance(sources, list):
            cleaned.append({"question": question, "answer": answer, "sources": sources})
    return cleaned


def _question_to_title(question: str) -> str:
    title = " ".join(question.strip().split())
    if len(title) > 42:
        title = title[:42] + "..."
    return title or "New chat"


def load_chat_store() -> dict:
    if not CHAT_HISTORY_FILE.exists():
        session = _new_session()
        return {"active_session_id": session["id"], "sessions": [session]}

    try:
        data = json.loads(CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        session = _new_session()
        return {"active_session_id": session["id"], "sessions": [session]}

    # Backward compatibility: old format stored a single history list.
    if isinstance(data, list):
        session = _new_session("Imported chat")
        session["history"] = _normalize_history(data)
        return {"active_session_id": session["id"], "sessions": [session]}

    if not isinstance(data, dict):
        session = _new_session()
        return {"active_session_id": session["id"], "sessions": [session]}

    raw_sessions = data.get("sessions", [])
    sessions: list[dict] = []
    if isinstance(raw_sessions, list):
        for raw in raw_sessions:
            if not isinstance(raw, dict):
                continue
            session_id = str(raw.get("id") or uuid4().hex[:12])
            title = str(raw.get("title") or "New chat")
            created_at = str(raw.get("created_at") or datetime.now(timezone.utc).isoformat())
            history = _normalize_history(raw.get("history", []))
            sessions.append(
                {
                    "id": session_id,
                    "title": title,
                    "created_at": created_at,
                    "history": history,
                }
            )

    if not sessions:
        session = _new_session()
        return {"active_session_id": session["id"], "sessions": [session]}

    active_session_id = str(data.get("active_session_id") or sessions[0]["id"])
    if active_session_id not in {s["id"] for s in sessions}:
        active_session_id = sessions[0]["id"]

    return {"active_session_id": active_session_id, "sessions": sessions}


def persist_chat_store(chat_store: dict) -> None:
    CHAT_HISTORY_FILE.write_text(json.dumps(chat_store, ensure_ascii=True), encoding="utf-8")


def get_active_session(chat_store: dict, active_session_id: str) -> dict:
    for session in chat_store["sessions"]:
        if session["id"] == active_session_id:
            return session
    return chat_store["sessions"][0]

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
if "chat_store" not in st.session_state:
    st.session_state.chat_store = load_chat_store()
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = st.session_state.chat_store["active_session_id"]

active_session = get_active_session(st.session_state.chat_store, st.session_state.active_session_id)
st.session_state.active_session_id = active_session["id"]
st.session_state.chat_store["active_session_id"] = active_session["id"]

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
    st.header("💬 Conversations")
    if st.button("＋ New chat", use_container_width=True):
        new_session = _new_session()
        st.session_state.chat_store["sessions"].insert(0, new_session)
        st.session_state.active_session_id = new_session["id"]
        st.session_state.chat_store["active_session_id"] = new_session["id"]
        persist_chat_store(st.session_state.chat_store)
        st.rerun()

    for session in st.session_state.chat_store["sessions"]:
        name_col, delete_col = st.columns([5, 1])
        label_prefix = "● " if session["id"] == st.session_state.active_session_id else ""
        label = f"{label_prefix}{session['title']}"
        with name_col:
            if st.button(label, key=f"open_{session['id']}", use_container_width=True):
                st.session_state.active_session_id = session["id"]
                st.session_state.chat_store["active_session_id"] = session["id"]
                persist_chat_store(st.session_state.chat_store)
                st.rerun()
        with delete_col:
            if st.button("🗑", key=f"delete_chat_{session['id']}", help="Delete this conversation"):
                sessions = st.session_state.chat_store["sessions"]
                if len(sessions) == 1:
                    sessions[0]["history"] = []
                    sessions[0]["title"] = "New chat"
                    st.session_state.active_session_id = sessions[0]["id"]
                else:
                    st.session_state.chat_store["sessions"] = [s for s in sessions if s["id"] != session["id"]]
                    if st.session_state.active_session_id == session["id"]:
                        st.session_state.active_session_id = st.session_state.chat_store["sessions"][0]["id"]
                st.session_state.chat_store["active_session_id"] = st.session_state.active_session_id
                persist_chat_store(st.session_state.chat_store)
                st.rerun()

    st.divider()
    if st.button("Clear current chat", use_container_width=True):
        active_session["history"] = []
        active_session["title"] = "New chat"
        persist_chat_store(st.session_state.chat_store)
        st.rerun()

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
# Chat transcript
# ---------------------------------------------------------------------------
for entry in active_session["history"]:
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

if not active_session["history"]:
    st.info("Upload documents in the sidebar and ask a question above to get started.")

# ---------------------------------------------------------------------------
# Chat input + query execution
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question about your documents...")

if question and question.strip():
    history_payload = []
    for turn in active_session["history"]:
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
            active_session["history"].append({
                "question": question.strip(),
                "answer": data["answer"],
                "sources": data["sources"],
            })
            if active_session["title"] == "New chat":
                active_session["title"] = _question_to_title(question)
            persist_chat_store(st.session_state.chat_store)
            st.rerun()
        except httpx.ConnectError:
            st.error("Cannot connect to the backend. Make sure the FastAPI server is running.")
        except Exception as e:
            st.error(f"Error: {e}")
