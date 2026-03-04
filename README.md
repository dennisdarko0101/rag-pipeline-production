# RAG Pipeline Production

[![CI](https://github.com/dennisdarko/rag-pipeline-production/actions/workflows/ci.yml/badge.svg)](https://github.com/dennisdarko/rag-pipeline-production/actions/workflows/ci.yml)
[![CD](https://github.com/dennisdarko/rag-pipeline-production/actions/workflows/cd.yml/badge.svg)](https://github.com/dennisdarko/rag-pipeline-production/actions/workflows/cd.yml)
[![Evaluation](https://github.com/dennisdarko/rag-pipeline-production/actions/workflows/eval.yml/badge.svg)](https://github.com/dennisdarko/rag-pipeline-production/actions/workflows/eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 272 passing](https://img.shields.io/badge/tests-272%20passing-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A production-grade Retrieval-Augmented Generation system built from scratch in Python. Combines hybrid search (semantic + BM25), cross-encoder reranking, and dual-LLM generation with automatic fallback to deliver accurate, citation-backed answers grounded in your documents.

## Architecture

```
  Streamlit UI (:8501)           FastAPI Backend (:8000)
 ┌──────────────────┐           ┌─────────────────────────────────────────────┐
 │  Chat | Eval     │   httpx   │                                             │
 │  Config | Status │ ────────▶ │  Ingest ──▶ Preprocess ──▶ Chunk ──▶ Embed  │
 │  Ingest          │           │                                       │     │
 └──────────────────┘           │                            ChromaDB ◀─┘     │
                                │                               │             │
                                │  Query ──▶ Hybrid Retriever ◀─┘             │
                                │              (Semantic + BM25, RRF)         │
                                │                    │                        │
                                │            Cross-Encoder Reranker           │
                                │                    │                        │
                                │            RAG Chain ──▶ FallbackLLM        │
                                │            (Claude ──▶ GPT-4o)              │
                                │                    │                        │
                                │            Citation Parser ──▶ RAGResponse  │
                                └─────────────────────────────────────────────┘
```

## Features

| Category | Highlights |
|----------|-----------|
| **Ingestion** | PDF, Markdown, text, web scraping. Recursive + semantic chunking. SHA-256 dedup. |
| **Embeddings** | OpenAI text-embedding-3-small with file-based cache (sharded, TTL, batch-aware). |
| **Retrieval** | Hybrid search (semantic + BM25) fused with Reciprocal Rank Fusion. Query expansion & HyDE. |
| **Reranking** | Cross-encoder (ms-marco-MiniLM) or LLM-based relevance scoring. Lazy model loading. |
| **Generation** | Dual-LLM (Claude + GPT-4o) with automatic fallback, citation validation, token tracking. |
| **Evaluation** | LLM-as-judge (4 metrics), 18-pair golden dataset, JSON/Markdown reports, CI quality gates. |
| **UI** | Streamlit dashboard with chat, evaluation, document ingestion, and live system status. |
| **API** | FastAPI with rate limiting, correlation IDs, structured logging, OpenAPI docs. |
| **DevOps** | Docker multi-stage builds, GitHub Actions CI/CD, 272 tests, pip-audit security scanning. |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/dennisdarko/rag-pipeline-production.git
cd rag-pipeline-production
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env    # Add your ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Seed and run
make seed               # Load sample docs into ChromaDB
make run                # API at http://localhost:8000
make run-ui             # UI at http://localhost:8501 (separate terminal)
```

### Docker (one command)

```bash
make docker-build && make docker-up
```

Starts API (`:8000`), ChromaDB (`:8001`), and Streamlit UI (`:8501`) with health check dependencies and network isolation.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Run the full RAG pipeline (retrieve, rerank, generate) |
| `POST` | `/api/v1/ingest` | Ingest a document from file path or URL |
| `POST` | `/api/v1/ingest/upload` | Upload and ingest a file (PDF, MD, TXT) |
| `POST` | `/api/v1/evaluate` | Evaluate RAG quality against Q&A pairs |
| `GET`  | `/health` | Component-level health check |

Interactive docs at `http://localhost:8000/docs`.

```bash
# Example: query the pipeline
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?", "k": 10, "rerank": true}'
```

## Streamlit Dashboard

The UI communicates with the backend exclusively via HTTP (no internal imports) and provides:

- **Chat tab** -- Conversational interface with expandable source cards and pipeline timing
- **Evaluation tab** -- Run golden dataset evaluation, view color-coded metric cards and per-question results
- **Sidebar** -- LLM provider selection, retrieval/reranking controls, system health, document ingestion
- **Dark theme** -- Professional slate/indigo palette with responsive layout

## Evaluation

Four LLM-as-judge metrics, each scored 0.0 -- 1.0 with explanations:

| Metric | What It Measures | CI Threshold |
|--------|-----------------|-------------|
| **Faithfulness** | Are claims supported by retrieved context? | >= 0.70 |
| **Answer Relevancy** | Does the answer address the question? | >= 0.70 |
| **Context Precision** | Are retrieved contexts relevant? | Monitored |
| **Context Recall** | Is needed information present in context? | Monitored |

Golden dataset: 18 Q&A pairs across 4 categories (straightforward, multi-chunk, unanswerable, adversarial). Weekly automated evaluation with GitHub issue creation on quality degradation.

```bash
make eval    # Run evaluation locally
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for full methodology.

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | Async HTTP server with OpenAPI docs |
| **Vector DB** | ChromaDB | Persistent vector storage with HNSW indexing |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim dense vectors with tiktoken counting |
| **LLM** | Claude 3.5 Sonnet + GPT-4o | Dual-LLM with automatic fallback |
| **Sparse Retrieval** | rank-bm25 | Okapi BM25 keyword matching |
| **Reranking** | sentence-transformers | Cross-encoder ms-marco-MiniLM-L-6-v2 |
| **Chunking** | LangChain text splitters | Recursive character + semantic chunking |
| **UI** | Streamlit | Interactive dashboard with dark theme |
| **Config** | Pydantic Settings | Type-safe configuration from `.env` |
| **Logging** | structlog | Structured JSON logging with context |
| **Testing** | pytest | 272 tests, 80%+ coverage required |
| **Linting** | ruff + mypy | Fast linting + strict type checking |
| **CI/CD** | GitHub Actions | Lint, test, security scan, Docker push |
| **Containers** | Docker + Compose | Multi-stage builds, 3-service stack |

## Project Structure

```
rag-pipeline-production/
├── src/
│   ├── api/                        # FastAPI application
│   │   ├── main.py                 #   App entry point, CORS, lifecycle
│   │   ├── schemas.py              #   Pydantic request/response models
│   │   ├── routes/                 #   query, ingest, evaluate, health
│   │   └── middleware/             #   Rate limiting, request logging
│   ├── ingestion/                  # Document loading, chunking, preprocessing
│   ├── embeddings/                 # OpenAI embedder + file-based cache
│   ├── vectorstore/                # ChromaDB implementation + ABC
│   ├── retrieval/                  # Semantic, BM25, Hybrid retrievers + rerankers
│   ├── generation/                 # LLM abstraction, RAG chain, citation parser
│   ├── evaluation/                 # Metrics, dataset, runner, report comparison
│   ├── config/                     # Pydantic Settings
│   ├── models/                     # Universal Document model
│   └── utils/                      # Structured logging, Prometheus metrics
├── tests/
│   ├── unit/                       # 256 unit tests (all APIs mocked)
│   ├── integration/                # 16 integration tests
│   └── eval/                       # Golden dataset (18 Q&A pairs)
├── ui/                             # Streamlit dashboard
│   ├── app.py                      #   Main app (chat + eval tabs)
│   ├── api_client.py               #   httpx client for backend
│   ├── components.py               #   Metric cards, source cards, timeline
│   └── config.py                   #   Theme, API URL, page settings
├── docker/
│   ├── Dockerfile                  # Multi-stage API build (non-root user)
│   ├── Dockerfile.ui               # Lightweight UI build
│   └── docker-compose.yml          # 3-service stack with network isolation
├── .github/workflows/
│   ├── ci.yml                      # Lint, type-check, test, security scan
│   ├── cd.yml                      # Docker build + push to ghcr.io
│   └── eval.yml                    # Scheduled evaluation + quality gates
├── docs/                           # Architecture, deployment, evaluation, handoff
├── scripts/                        # setup.sh, seed_db.sh, run_eval.sh
├── data/sample_docs/               # 4 technical articles for testing
├── pyproject.toml                  # Dependencies and tool configuration
├── Makefile                        # All development commands
├── .dockerignore                   # Docker build exclusions
├── CONTRIBUTING.md                 # Contributing guidelines
└── LICENSE                         # MIT
```

## Testing

```bash
make test           # Run all 272 tests
make test-cov       # Run with HTML coverage report (80%+ required)
make eval           # Run RAG evaluation against golden dataset
```

All external APIs (OpenAI, Anthropic, ChromaDB) are mocked -- no API keys needed for the test suite.

**Breakdown:** 256 unit tests + 16 integration tests covering loaders, chunkers, embedder, cache, retrievers, rerankers, LLM, prompts, chain, API endpoints, evaluation metrics/dataset/runner, and full API request/response cycles.

## CI/CD

| Workflow | Trigger | Jobs |
|----------|---------|------|
| **CI** | Push/PR to `main` | Lint, type-check, test (Python 3.11 + 3.12), security scan |
| **CD** | Push to `main` | Build + push Docker images to `ghcr.io` |
| **Eval** | Weekly + manual | RAG quality evaluation with threshold checks, auto-creates issues on degradation |

## Configuration

All settings via environment variables (`.env`), loaded through Pydantic Settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | | Claude API key |
| `OPENAI_API_KEY` | | OpenAI API key (embeddings + GPT-4o fallback) |
| `LLM_MODEL` | `claude-3-5-sonnet-20241022` | Primary LLM |
| `LLM_FALLBACK_MODEL` | `gpt-4o` | Fallback LLM |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `RETRIEVAL_TOP_K` | `10` | Documents to retrieve |
| `RERANK_TOP_K` | `5` | Documents after reranking |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `RATE_LIMIT_REQUESTS` | `60` | Requests per rate-limit window |
| `LOG_LEVEL` | `INFO` | Logging level |

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the complete reference.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) -- System design, data flow diagrams, design decisions
- [Deployment](docs/DEPLOYMENT.md) -- Local setup, Docker, environment variables, cloud guide
- [Evaluation](docs/EVALUATION.md) -- Metrics, golden dataset, CI integration, Python API
- [Handoff](docs/HANDOFF.md) -- Complete project status, file inventory, architecture decisions

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE)
