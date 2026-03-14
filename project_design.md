# SG-ComplianceGuard 🇸🇬

**SG-ComplianceGuard** is a privacy-first, local-LLM analysis tool designed to help users navigate complex Singaporean government regulations (starting with the Ministry of Manpower). 

Unlike standard AI wrappers, this project utilizes a **Local RAG (Retrieval-Augmented Generation)** architecture to ensure that sensitive employment data never leaves the user's machine, ensuring full alignment with **PDPA (Personal Data Protection Act)** principles.

---

## 🚀 Key Features

* **Automated Ingestion:** Uses **Crawl4AI** to monitor and scrape official MOM sources, converting messy HTML tables and nested lists into LLM-optimized Markdown.
* **Side-by-Side Verification:** A custom **Streamlit** interface that allows users to verify AI-generated answers against the original official source text in real-time.
* **Local-First Intelligence:** Powered by **Ollama**, allowing the use of state-of-the-art models (Llama 3/Mistral) without internet dependency or API costs.
* **Hybrid Search:** Implements a vector-based search using **Qdrant/ChromaDB** combined with keyword matching to ensure high-precision retrieval of legal statutes.

---

## 🛠️ Tech Stack

| Layer               | Technology                                                                 |
|---------------------|----------------------------------------------------------------------------|
| **Inference Engine**| [Ollama](https://ollama.ai/) (Llama 3.2 / Mistral)                         |
| **Data Ingestion** | [Crawl4AI](https://github.com/unclecode/crawl4ai)                          |
| **Vector Database** | [Qdrant](https://qdrant.tech/) (Dockerized)                                |
| **Frontend** | [Streamlit](https://streamlit.io/)                                         |
| **Orchestration** | [Docker](https://www.docker.com/) & Docker Compose                         |
| **Framework** | [FastAPI](https://fastapi.tiangolo.com/) & LangChain/LlamaIndex             |

---

## 🏗️ Architecture

1.  **Crawler Service:** Periodically fetches updates from MOM FAQ pages and converts them to clean Markdown.
2.  **Embedding Pipeline:** Text is chunked using recursive character splitting and embedded via Hugging Face local models.
3.  **Vector Store:** Vectors are stored in a persistent Qdrant volume within the Docker environment.
4.  **UI Layer:** Streamlit queries the FastAPI backend, which orchestrates the retrieval from Qdrant and the generation via Ollama.

---

## 📦 Installation & Setup

This project is fully containerized. To run it locally, ensure you have **Docker** and **Ollama** installed.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/sg-compliance-guard.git](https://github.com/your-username/sg-compliance-guard.git)
   cd sg-compliance-guard