# Retrieval-Augmented Generation Engine & Evaluation Pipeline

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://github.com/Daniel-Downes/domain-rag-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Daniel-Downes/RAG_Engine/actions)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg?logo=githubactions&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Features

- **Hybrid Search Engine:** Combines dense vector similarity (`all-MiniLM-L6-v2`) with sparse keyword matching (`BM25`) and reranking to maximize context relevance.
- **Asynchronous Streaming API:** High-throughput FastAPI backend utilizing Server-Sent Events (SSE) for low-latency, real-time response generation.
- **Automated MLOps Benchmarking:** Built-in integration with `Ragas` to evaluate **Faithfulness**, **Answer Relevance**, **Context Recall**, and **Context Precision**.
- **Interactive UI:** Dynamic frontend displaying inline markdown citations and expandable source-chunk metadata.
- **Containerized Architecture:** Fully dockerized services (`Docker Compose`) backed by automated CI/CD via **GitHub Actions**.

---

## Architecture Overview

```text
[ Document Ingestion ]
       │
       ├──► Recursive Semantic Chunker
       └──► Embeddings Engine (Sentence-Transformers)
                 │
                 ▼
          [ Qdrant Vector DB ]
                 │
[ User Query ] ──┼──► Hybrid Search (Dense + BM25) ──► Cohere Reranker
                                                              │
                                                              ▼
[ React Frontend ] ◄── Server-Sent Events ◄── FastAPI ◄── LLM Synthesis Engine
                                │
                                ▼
                   [ Ragas Evaluation Suite ]
```

---

## Tech Stack

| Domain  | Technologies Used | 
|----------|-------|
| Backend & API   | Python 3.13, FastAPI, Uvicorn, Pydantic  | 
| Vector Database & Search    | Qdrant, RankBM25, LangChain  | 
| Embeddings & LLMs  | Hugging Face (sentence-transformers), OpenAI API / Ollama   | 
| Evaluation Framework  | Ragas, TruLens   | 
| DevOps & Infrastructure  | Docker, Docker Compose, GitHub Actions, Pytest, Ruff   | 

---

## Getting Started
**Prerequisites**
- Docker Desktop installed
- Python 3.13+
- OpenAI API Key (or local Ollama instance)

---

## Local Setup & Insallation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Daniel-Downes/RAG_Engine.git
    cd RAG_Engine
    ```
2. **Configure environment variables:**
    
    Copy the example environment file and add your credentials

    ```bash
    cp .env.example .env
    ```
    *Edit ```.env``` to include your configuration:*
    ```bash
    **Code snippet**

    QDRANT_HOST=localhost
    QDRANT_PORT=6333
    OPENAI_API_KEY=your_openai-api-key-here
    EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
    ```
3. **Start up vector database infrastructure:**
    ```bash
    docker compose up -d
    ```
4. **Set up Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
5. **Ingest documents & launch application:**
    ```bash
    # Add your source files (PDFs/Markdown) to data/raw_docs/
    python -m src.vectorstore.indexer

    # Start the FastAPI server
    uvicorn src.main:app --reload
    ```
    Navigate to ```http://localhost:8000/docs``` to view the interactive Swagger API documentation.

## Automated Evaluation & Metrics

To prevent model hallucinations and maintain context precision across code updates, run the evaluation suite:
```bash
python -m test.eval_ragas
```
## Repository Structure
```text
RAG-engine/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI pipeline
├── data/
│   └── raw_docs/              # Ingestion directory for target PDFs
├── src/
│   ├── ingestion/             # PDF extraction & semantic chunking
│   ├── vectorstore/           # Qdrant client connection & indexing pipelines
│   ├── evaluation/            # Ragas benchmark execution scripts
│   └── main.py                # FastAPI entry point & SSE streaming
├── tests/                     # Unit and integration test suites
├── docker-compose.yml         # Container configuration for Qdrant
├── requirements.txt           # Environment dependency constraints
└── README.md
```

## License
Distributed under the MIT License. See ```License``` for details.