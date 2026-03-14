# SG-ComplianceGuard

A privacy-first, local RAG (Retrieval-Augmented Generation) tool for navigating Singapore Ministry of Manpower (MOM) regulations. All data stays on your machine — no external LLM APIs, no data leakage, fully aligned with PDPA principles.

---

## How It Works

### Data Flow

```
MOM Website (25 pages)
    │
    │  Crawl4AI + Playwright/Chromium
    ▼
data/raw_markdown/*.md          ← one file per MOM page, with source_url + scraped_at frontmatter
    │
    │  RecursiveCharacterTextSplitter (512 chars, 64 overlap)
    ▼
555 text chunks
    │
    │  all-MiniLM-L6-v2 (HuggingFace, runs locally via sentence-transformers)
    ▼
555 vectors (384-dim)
    │
    │  qdrant-client (HTTP to Docker container)
    ▼
Qdrant collection: "mom_regulations"   ← each point stores vector + text + source_url
    │
    │  User submits a query via Streamlit
    ▼
FastAPI /query endpoint
    ├── embed query → same all-MiniLM-L6-v2 model
    ├── vector search → top-k matching chunks from Qdrant
    └── prompt + context → Ollama (llama3.2:3b, runs locally)
    │
    ▼
Streamlit UI
    ├── AI-generated answer
    └── source citations (original MOM page URLs)
```

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Web Crawling** | [Crawl4AI](https://github.com/unclecode/crawl4ai) v0.8.0 + Playwright/Chromium | Scrapes MOM FAQ pages, renders JS, converts HTML to clean Markdown |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` | Splits markdown into 512-char overlapping chunks for embedding |
| **Embedding Model** | `all-MiniLM-L6-v2` via [sentence-transformers](https://www.sbert.net/) | Converts text chunks into 384-dim semantic vectors, runs fully in-process |
| **Vector Database** | [Qdrant](https://qdrant.tech/) (Dockerized) | Stores and searches vectors with cosine similarity; payloads include source URL for citation |
| **LLM Inference** | [Ollama](https://ollama.ai/) — `llama3.2:3b` | Runs LLM locally, generates answers from retrieved context chunks |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) | Orchestrates query embedding → Qdrant retrieval → Ollama generation |
| **Frontend** | [Streamlit](https://streamlit.io/) | Chat UI with side-by-side source verification panel |
| **Orchestration** | Docker & Docker Compose | Runs Qdrant, FastAPI, and Streamlit as isolated services |

---

## Project Structure

```
rag_lab/
├── crawler/
│   ├── config.py          # MOM target URLs (25 pages)
│   ├── scraper.py         # Crawl4AI async scraper → data/raw_markdown/
│   ├── requirements.txt
│   └── Dockerfile
├── backend/
│   ├── config.py          # Shared config (Qdrant, Ollama, model, chunk sizes)
│   ├── embedder.py        # Chunking + embedding + Qdrant ingestion pipeline
│   ├── main.py            # FastAPI app with /query endpoint
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py             # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   └── raw_markdown/      # Scraped .md files (gitignored, regenerable)
├── docker-compose.yml     # Qdrant + backend + frontend + crawler services
└── steps.md               # Setup progress tracker
```

---

## Setup

### Prerequisites
- Docker Desktop
- Ollama with `llama3.2:3b` pulled (`ollama pull llama3.2:3b`)
- Python 3.11+ with a virtual environment

### 1. Scrape MOM pages
```bash
cd crawler
pip install -r requirements.txt
crawl4ai-setup && playwright install chromium
python scraper.py
# → writes 25 .md files to data/raw_markdown/
```

### 2. Start Qdrant
```bash
docker compose up qdrant -d
```

### 3. Embed and index
```bash
pip install -r backend/requirements.txt
python backend/embedder.py
# → chunks 25 docs into 555 vectors, loads into Qdrant
```

### 4. Start the full stack
```bash
docker compose up
```

Visit `http://localhost:8501` for the Streamlit UI.
Visit `http://localhost:6333/dashboard` to inspect the Qdrant collection.

---

## Privacy

- No data ever leaves your machine
- Embedding runs in-process (no HuggingFace API calls after first model download)
- LLM inference via Ollama runs fully locally
- Qdrant runs in a local Docker container
