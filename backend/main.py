"""
FastAPI backend — Section 6.

POST /query  →  embed query → hybrid search (vector + keyword) → Ollama LLM → response + sources
GET  /health →  liveness check
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, Query

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_PORT,
    OLLAMA_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOP_K = 5          # number of results from each search leg
OLLAMA_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# Shared state (loaded once on startup)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    _state["embedder"] = SentenceTransformer(EMBED_MODEL)
    _state["qdrant"] = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("Backend ready.")
    yield
    _state.clear()


app = FastAPI(title="SG-ComplianceGuard API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    text: str
    source_url: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _embed(text: str) -> list[float]:
    return _state["embedder"].encode(text).tolist()


def _vector_search(vector: list[float], limit: int) -> list[dict]:
    results = _state["qdrant"].query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
        with_payload=True,
    ).points
    return [
        {"id": r.id, "score": r.score, "payload": r.payload}
        for r in results
    ]


def _keyword_search(query: str, limit: int) -> list[dict]:
    """Full-text search against the indexed 'text' payload field."""
    results = _state["qdrant"].scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="text",
                    match=MatchText(text=query),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {"id": r.id, "score": 0.0, "payload": r.payload}
        for r in results[0]
    ]


def _reciprocal_rank_fusion(
    vector_hits: list[dict],
    keyword_hits: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine vector and keyword results using Reciprocal Rank Fusion.
    Higher RRF score = better combined rank.
    """
    scores: dict[int, float] = {}
    payloads: dict[int, dict] = {}

    for rank, hit in enumerate(vector_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        payloads[doc_id] = hit["payload"]

    for rank, hit in enumerate(keyword_hits):
        doc_id = hit["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        payloads[doc_id] = hit["payload"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"id": doc_id, "score": score, "payload": payloads[doc_id]}
        for doc_id, score in ranked[:TOP_K]
    ]


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['payload']['source_url']}]\n{c['payload']['text']}"
        for c in context_chunks
    )
    return f"""You are a helpful Singapore employment law assistant. 
Answer the question using ONLY the context below. 
If the answer is not in the context, say "I don't have enough information to answer that."
Be concise and cite which source your answer is based on.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


async def _call_ollama(prompt: str) -> str:
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL, "collection": COLLECTION_NAME}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Query: {req.question!r}")

    # 1. Embed the query
    vector = _embed(req.question)

    # 2. Hybrid search — vector + keyword, fused with RRF
    vector_hits = _vector_search(vector, limit=TOP_K * 2)
    keyword_hits = _keyword_search(req.question, limit=TOP_K * 2)
    fused = _reciprocal_rank_fusion(vector_hits, keyword_hits)

    logger.info(f"Retrieved {len(fused)} chunks after fusion")

    # 3. Build prompt and call Ollama
    prompt = _build_prompt(req.question, fused)
    answer = await _call_ollama(prompt)

    # 4. Build source list (deduplicated by URL)
    seen_urls: set[str] = set()
    sources = []
    for hit in fused:
        url = hit["payload"].get("source_url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append(SourceDoc(
                text=hit["payload"]["text"],
                source_url=url,
                score=round(hit["score"], 4),
            ))

    return QueryResponse(answer=answer, sources=sources)
