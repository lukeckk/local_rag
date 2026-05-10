import os

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "document_rag"

# Embedding model (HuggingFace sentence-transformers)
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # output dimension for all-MiniLM-L6-v2

# Text splitting
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
