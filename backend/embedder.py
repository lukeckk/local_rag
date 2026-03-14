"""
Embedding pipeline — Section 4.

Reads scraped .md files from data/raw_markdown/, chunks them using
RecursiveCharacterTextSplitter, embeds each chunk with all-MiniLM-L6-v2,
and upserts into a Qdrant collection.

Run once after scraper.py to populate the vector store:
    python embedder.py
"""

import os
import re
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBED_MODEL,
    EMBED_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAW_MARKDOWN_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body. Returns (metadata, body)."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()

    return meta, parts[2].strip()


def load_documents(raw_dir: Path) -> list[dict]:
    """Load all .md files, returning a list of {source_url, scraped_at, content} dicts."""
    docs = []
    for path in sorted(raw_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        if not body:
            logger.warning(f"Skipping {path.name}: empty body after frontmatter")
            continue
        docs.append({
            "source_url": meta.get("source_url", path.stem),
            "scraped_at": meta.get("scraped_at", ""),
            "content": body,
            "filename": path.name,
        })
    logger.info(f"Loaded {len(docs)} documents from {raw_dir}")
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split each document into overlapping chunks, preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for i, text in enumerate(splits):
            chunks.append({
                "text": text,
                "source_url": doc["source_url"],
                "scraped_at": doc["scraped_at"],
                "filename": doc["filename"],
                "chunk_index": i,
            })
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed each chunk's text using all-MiniLM-L6-v2."""
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks…")
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True)

    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector.tolist()

    logger.info("Embedding complete.")
    return chunks


def upsert_to_qdrant(chunks: list[dict]) -> None:
    """Create (or recreate) the Qdrant collection and upsert all embedded chunks."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists — recreating.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    logger.info(f"Collection '{COLLECTION_NAME}' created (dim={EMBED_DIM}, cosine).")

    points = [
        PointStruct(
            id=i,
            vector=chunk["vector"],
            payload={
                "text": chunk["text"],
                "source_url": chunk["source_url"],
                "scraped_at": chunk["scraped_at"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"Upserted points {start}–{start + len(batch) - 1}")

    # Full-text index on the text payload field — enables keyword search in hybrid mode
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text",
        field_schema="text",
    )
    logger.info("Full-text index created on 'text' field.")

    count = client.count(collection_name=COLLECTION_NAME).count
    logger.info(f"Done. {count} points now in '{COLLECTION_NAME}'.")


def run_pipeline() -> None:
    raw_dir = Path(os.path.join(os.path.dirname(__file__), RAW_MARKDOWN_DIR))
    if not raw_dir.exists():
        logger.error(f"raw_markdown directory not found: {raw_dir}")
        return

    docs = load_documents(raw_dir)
    if not docs:
        logger.error("No documents found. Run scraper.py first.")
        return

    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks)
    upsert_to_qdrant(chunks)


if __name__ == "__main__":
    run_pipeline()
