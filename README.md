# LOCAL RAG

LOCAL RAG is a local-first Retrieval-Augmented Generation application for asking questions over uploaded documents.  
It runs fully on your machine using Docker services and Ollama for inference.

## What this project does

- Upload and index documents: PDF, CSV, XLSX, XLS, and TXT
- Store embeddings in Qdrant vector database
- Run hybrid retrieval (vector + keyword)
- Answer with local LLM inference through Ollama
- Show source chunks used for answers
- Persist chat sessions in the UI across refresh/restart
- Let users inspect and delete indexed files from the vector store

## Stack

- Frontend: Streamlit
- Backend: FastAPI
- Vector DB: Qdrant
- Embeddings: `all-MiniLM-L6-v2`
- LLM runtime: Ollama (`llama3.2:3b`)
- Orchestration: Docker Compose

## Project structure

```text
backend/        FastAPI API, retrieval logic, indexing logic
frontend/       Streamlit chat UI
homepage/       Static setup page for end users
docker-compose.yml
```

## Prerequisites

- Docker Desktop
- Ollama
- Model pulled locally:

```bash
ollama pull llama3.2:3b
```

## Quick start

```bash
git clone <your-repo-url>
cd rag_documents
docker compose up -d --build
```

## Service URLs

- Frontend: `http://localhost:8502`
- Backend health: `http://localhost:8001/health`
- Backend docs: `http://localhost:8001/docs`
- Qdrant dashboard: `http://localhost:6335/dashboard`

## Usage

1. Open the frontend.
2. Upload one or more files in the sidebar.
3. Click `Process & Index`.
4. Ask questions in chat.
5. Expand `Sources` to inspect supporting context.
6. Use `Indexed Files` to remove document chunks from Qdrant.

## Notes on retrieval behavior

- Rank values shown in sources are retrieval rank scores, not answer confidence.
- CSV/Excel ingestion is row-oriented to improve record-level lookup.
- Conversation is stateful per chat session in the UI.

## Reset local data

Remove containers and Qdrant volume:

```bash
docker compose down -v
```

## License

Add your preferred license information here.
