import logging
from contextlib import asynccontextmanager
from typing import Any, Literal
import re
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue
import shutil
from pathlib import Path

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_HOST,
    OLLAMA_PORT,
    OLLAMA_MODEL,
)
from embedder import DocumentProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOP_K = 5
OLLAMA_TIMEOUT = 120.0
MIN_VECTOR_SCORE = 0.18
MAX_HISTORY_MESSAGES = 1000
MAX_HISTORY_MESSAGES_IN_PROMPT = 40
TABLE_CONTEXT_LIMIT = 30

# Global state
_state: dict[str, Any] = {}
_processor = DocumentProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    _state["embedder"] = _processor.embedder
    _state["qdrant"] = _processor.client
    logger.info("Backend ready.")
    yield
    _state.clear()

app = FastAPI(title="Document RAG API", lifespan=lifespan)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_path = Path(f"/tmp/{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        num_chunks = _processor.process_and_index(temp_path, file.filename)
        return {"status": "success", "filename": file.filename, "chunks": num_chunks}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)

class SourceDoc(BaseModel):
    text: str
    source_url: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]


class DocumentSummary(BaseModel):
    filename: str
    chunks: int

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


def _extract_purchase_lookup(question: str) -> tuple[str | None, str | None]:
    year_match = re.search(r"\b(20\d{2})\b", question)
    year = year_match.group(1) if year_match else None

    name_patterns = [
        r"what did (.+?) purchase",
        r"purchases? by (.+?)(?: in|$)",
        r"what has (.+?) purchased",
    ]
    candidate_name = None
    for pattern in name_patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            candidate_name = match.group(1).strip(" ?!.,")
            break

    if not candidate_name:
        return None, year

    words = [w for w in candidate_name.split() if w]
    if len(words) < 2 or len(words) > 4:
        return None, year
    normalized_name = " ".join(w.capitalize() for w in words)
    return normalized_name, year


def _table_purchase_lookup(customer_name: str, year: str | None) -> list[dict]:
    must_conditions = [
        FieldCondition(key="text", match=MatchText(text=f"customer name: {customer_name}"))
    ]
    if year:
        must_conditions.append(
            FieldCondition(key="text", match=MatchText(text=f"date: {year}-"))
        )

    offset = None
    collected: list[dict] = []
    while True:
        points, next_offset = _state["qdrant"].scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=must_conditions),
            limit=128,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            collected.append({"id": point.id, "score": 1.0, "payload": point.payload or {}})
        if next_offset is None:
            break
        offset = next_offset

    def row_number(hit: dict) -> int:
        text = hit.get("payload", {}).get("text", "")
        row_match = re.search(r"Row\s+(\d+)\s+\|", text)
        return int(row_match.group(1)) if row_match else 10**9

    collected.sort(key=row_number)
    return collected


def _parse_table_row(text: str) -> dict[str, str]:
    record: dict[str, str] = {}
    for part in [p.strip() for p in text.split("|")]:
        row_match = re.match(r"Row\s+(\d+)$", part)
        if row_match:
            record["row"] = row_match.group(1)
            continue
        if ":" in part:
            key, value = part.split(":", 1)
            record[key.strip().lower()] = value.strip()
    return record


def _build_purchase_answer(customer_name: str, year: str | None, hits: list[dict]) -> str:
    rows = []
    for hit in hits:
        record = _parse_table_row(hit.get("payload", {}).get("text", ""))
        if record:
            rows.append(record)

    if not rows:
        scope = f"in {year}" if year else "in the indexed data"
        return f"I couldn't find purchases for {customer_name} {scope}."

    year_text = f" in {year}" if year else ""
    lines = [f"{customer_name} made {len(rows)} purchase(s){year_text}:"]
    for row in rows:
        date = row.get("date", "unknown date")
        product = row.get("product", "unknown product")
        price = row.get("price", "unknown price")
        row_no = row.get("row", "?")
        lines.append(f"- {date}: {product} (${price}) [Row {row_no}]")
    return "\n".join(lines)


def _list_documents() -> list[DocumentSummary]:
    counts: dict[str, int] = {}
    offset = None

    while True:
        points, next_offset = _state["qdrant"].scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            filename = payload.get("filename") or payload.get("source_url") or "unknown"
            counts[str(filename)] = counts.get(str(filename), 0) + 1

        if next_offset is None:
            break
        offset = next_offset

    return [
        DocumentSummary(filename=name, chunks=count)
        for name, count in sorted(counts.items())
    ]

def _reciprocal_rank_fusion(
    vector_hits: list[dict],
    keyword_hits: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(vector_hits):
        doc_id = str(hit["id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        payloads[doc_id] = hit["payload"]

    for rank, hit in enumerate(keyword_hits):
        doc_id = str(hit["id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        payloads[doc_id] = hit["payload"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"id": doc_id, "score": score, "payload": payloads[doc_id]}
        for doc_id, score in ranked[:TOP_K]
    ]

def _build_prompt(question: str, context_chunks: list[dict], history: list[ChatMessage]) -> str:
    history_for_prompt = history[-MAX_HISTORY_MESSAGES_IN_PROMPT:]
    conversation = "\n".join(
        f"{msg.role.upper()}: {msg.content}"
        for msg in history_for_prompt
    )
    context = "\n\n---\n\n".join(
        f"[Source: {c['payload']['source_url']}]\n{c['payload']['text']}"
        for c in context_chunks
    ) or "(No retrieved document context)"
    return f"""You are a helpful assistant.
If relevant context is provided below, use it as the primary source of truth.
If context is missing, insufficient, or not relevant to the question, answer naturally using your general knowledge.
If the user sends a greeting or casual small talk, respond conversationally and do not say you lack enough information.
When you use context, cite the supporting source label(s). When you do not use context, do not invent citations.

CONVERSATION HISTORY:
{conversation}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def _should_skip_retrieval(question: str) -> bool:
    normalized = question.strip().lower()
    words = normalized.split()
    if not words:
        return True
    if normalized in {"hi", "hello", "hey", "yo", "sup", "thanks", "thank you"}:
        return True
    # Very short conversational turns are usually chat, not document QA.
    return len(words) <= 2 and not normalized.endswith("?")


def _is_profile_query(question: str) -> bool:
    normalized = question.strip().lower()
    profile_phrases = (
        "who am i",
        "about me",
        "my resume",
        "my cv",
        "my background",
        "my experience",
        "summarize me",
    )
    return any(phrase in normalized for phrase in profile_phrases)


async def _call_ollama(prompt: str, system_prompt: str | None = None) -> str:
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt
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


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents():
    return _list_documents()


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    _state["qdrant"].delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
        ),
        wait=True,
    )
    return {"status": "success", "filename": filename}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(req.history) > MAX_HISTORY_MESSAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long. Maximum supported history is {MAX_HISTORY_MESSAGES} messages.",
        )

    logger.info(f"Query: {req.question!r}")

    table_lookup_name, table_lookup_year = _extract_purchase_lookup(req.question)

    if _should_skip_retrieval(req.question):
        vector_hits = []
        fused = []
    else:
        table_hits = []
        if table_lookup_name:
            table_hits = _table_purchase_lookup(table_lookup_name, table_lookup_year)

        if table_hits:
            vector_hits = []
            fused = table_hits[:TABLE_CONTEXT_LIMIT]
            logger.info(
                f"Using table lookup context for customer={table_lookup_name!r}, year={table_lookup_year!r}, hits={len(fused)}"
            )
        else:
            vector = _embed(req.question)
            vector_hits = _vector_search(vector, limit=TOP_K * 2)
            keyword_hits = _keyword_search(req.question, limit=TOP_K * 2)
            fused = _reciprocal_rank_fusion(vector_hits, keyword_hits)

    logger.info(f"Retrieved {len(fused)} chunks after fusion")

    forced_table_context = bool(table_lookup_name and fused)
    top_vector_score = vector_hits[0]["score"] if vector_hits else 0.0
    has_relevant_context = bool(fused) and (
        forced_table_context or top_vector_score >= MIN_VECTOR_SCORE or _is_profile_query(req.question)
    )

    prompt_context = fused if has_relevant_context else []
    if forced_table_context and table_lookup_name:
        sources = [
            SourceDoc(
                text=hit["payload"].get("text", ""),
                source_url=hit["payload"].get("source_url", ""),
                score=round(hit.get("score", 1.0), 4),
            )
            for hit in fused
        ]
        answer = _build_purchase_answer(table_lookup_name, table_lookup_year, fused)
        return QueryResponse(answer=answer, sources=sources)

    prompt = _build_prompt(req.question, prompt_context, req.history)
    system_prompt = (
        "You are a document assistant. You DO have access to the provided CONTEXT block. "
        "Never say you cannot access files when context is present. Prefer the provided context, "
        "be accurate, and cite sources when using it."
        if has_relevant_context
        else "You are a friendly conversational assistant. Reply naturally to greetings and normal chat."
    )
    answer = await _call_ollama(prompt, system_prompt=system_prompt)

    sources = []
    if has_relevant_context:
        seen_source_blocks: set[tuple[str, str]] = set()
        for hit in fused:
            url = hit["payload"].get("source_url", "")
            text = hit["payload"].get("text", "")
            source_key = (url, text)
            if source_key in seen_source_blocks:
                continue
            seen_source_blocks.add(source_key)
            sources.append(SourceDoc(
                text=text,
                source_url=url,
                score=round(hit["score"], 4),
            ))

    return QueryResponse(answer=answer, sources=sources)
