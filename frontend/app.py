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
    page_title="SG-ComplianceGuard",
    page_icon="🇸🇬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🇸🇬 SG-ComplianceGuard")
st.caption(
    "Ask questions about Singapore MOM employment regulations. "
    "All answers are grounded in official MOM sources — no hallucination, no data leaves your machine."
)
st.divider()

# ---------------------------------------------------------------------------
# Session state — preserve conversation history + input state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "auto_submit" not in st.session_state:
    st.session_state.auto_submit = False

# ---------------------------------------------------------------------------
# Sidebar — example questions + info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Example Questions")
    example_questions = [
        "What is the minimum salary for an Employment Pass?",
        "How many days of sick leave is an employee entitled to?",
        "What are the grounds for wrongful dismissal?",
        "What is the notice period for termination?",
        "Who is covered under the Employment Act?",
        "What are the S Pass quota requirements?",
        "Can an employer deduct salary without consent?",
        "What happens to unused annual leave upon resignation?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.current_question = q
            st.session_state.auto_submit = True
            st.rerun()

    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- 🔍 Qdrant (vector DB)\n"
        "- 🤗 all-MiniLM-L6-v2 (embeddings)\n"
        "- 🦙 llama3.2:3b via Ollama\n"
        "- ⚡ FastAPI backend\n"
        "- 🔒 100% local — PDPA compliant"
    )

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
question = st.text_input(
    "Ask a question about MOM regulations:",
    value=st.session_state.current_question,
    placeholder="e.g. What is the minimum salary for an Employment Pass?",
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

# Trigger from sidebar example button
if st.session_state.auto_submit:
    st.session_state.auto_submit = False
    ask_clicked = True
    question = st.session_state.current_question

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
if ask_clicked and question.strip():
    st.session_state.current_question = question.strip()
    with st.spinner("Searching MOM regulations and generating answer…"):
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

    # Left — AI answer
    with answer_col:
        st.markdown("#### 🤖 AI Answer")
        st.markdown(entry["answer"])

    # Right — source verification panel
    with source_col:
        st.markdown("#### 📄 Source Verification")
        st.caption("Original MOM text the answer is based on:")

        for i, src in enumerate(entry["sources"]):
            with st.expander(
                f"Source {i + 1} — {src['source_url'].split('/')[-1].replace('-', ' ').title()}",
                expanded=(i == 0),
            ):
                st.markdown(
                    f"**[View on MOM website ↗]({src['source_url']})**",
                    unsafe_allow_html=True,
                )
                st.caption(f"Relevance score: {src['score']:.4f}")
                st.markdown("---")
                st.markdown(src["text"])

    st.divider()

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
if not st.session_state.history:
    st.info("Ask a question above to get started. Use the sidebar for examples.")
