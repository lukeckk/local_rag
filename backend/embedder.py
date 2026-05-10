import os
import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
import pypdf

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    EMBED_MODEL,
    EMBED_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

class DocumentProcessor:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="text",
                field_schema="text",
            )

    def extract_text(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext == ".txt":
            return file_path.read_text(encoding="utf-8")
        elif ext == ".pdf":
            text = ""
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        elif ext in [".csv", ".xlsx", ".xls"]:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            return self._table_to_text(df)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _table_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert tabular data to row-wise natural-language lines.
        This retrieval-friendly format preserves column semantics better
        than DataFrame.to_string() for RAG over CSV/Excel.
        """
        if df.empty:
            return ""

        normalized = df.fillna("")
        lines: list[str] = []
        for i, row in normalized.iterrows():
            parts: list[str] = []
            for col, value in row.items():
                col_name = str(col).strip()
                val_text = str(value).strip()
                parts.append(f"{col_name}: {val_text}")
            lines.append(f"Row {i + 1} | " + " | ".join(parts))
        return "\n".join(lines)

    def process_and_index(self, file_path: Path, filename: str):
        content = self.extract_text(file_path)
        ext = file_path.suffix.lower()
        if ext in [".csv", ".xlsx", ".xls"]:
            # Keep each table row as an independent chunk to avoid
            # mixing neighboring records during retrieval.
            splits = [line for line in content.splitlines() if line.strip()]
        else:
            splits = self.splitter.split_text(content)
        
        points = []
        for i, text in enumerate(splits):
            vector = self.embedder.encode(text).tolist()
            points.append(PointStruct(
                id=os.urandom(16).hex(),
                vector=vector,
                payload={
                    "text": text,
                    "source_url": filename,
                    "filename": filename,
                    "chunk_index": i,
                }
            ))
        
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)
