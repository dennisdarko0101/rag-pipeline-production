# RAG Pipeline Production

A production-grade Retrieval-Augmented Generation system built from scratch in Python. Combines hybrid search (semantic + BM25), cross-encoder reranking, and dual-LLM generation with automatic fallback to deliver accurate, citation-backed answers grounded in your documents.

**166 tests passing | Full CI/CD | Docker-ready | Structured logging throughout**

## Architecture

```
                            RAG Pipeline Architecture
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │  ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌──────────────────────┐  │
 │  │  Loader   │──▶│ Preprocessor│──▶│  Chunker  │──▶│  OpenAI Embedder    │  │
 │  │ PDF/MD/   │   │ Clean/Dedup│   │ Recursive │   │  + Embedding Cache   │  │
 │  │ Text/Web  │   │ Fingerprint│   │ Semantic  │   │  (file-based, TTL)   │  │
 │  └──────────┘   └────────────┘   └───────────┘   └──────────┬───────────┘  │
 │                                                              │              │
 │                          INGESTION                           ▼              │
 │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ┌──────────────────┐   │
 │                                                     │  ChromaDB Vector │   │
 │                                                     │  Store (cosine)  │   │
 │                          STORAGE                    └────────┬─────────┘   │
 │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─          │              │
 │                                                              │              │
 │  ┌─────────┐   ┌────────────────┐   ┌────────────┐         │              │
 │  │  User   │──▶│ Hybrid Retriever│◀──│  BM25      │         │              │
 │  │  Query  │   │ (RRF Fusion)   │◀──│  Retriever │         │              │
 │  └─────────┘   │ w=0.7 / w=0.3 │   └────────────┘         │              │
 │                 │                │◀──┌────────────┐         │              │
 │                 └───────┬────────┘   │  Semantic  │◀────────┘              │
 │                         │            │  Retriever │                        │
 │                         ▼            └────────────┘                        │
 │                 ┌───────────────┐                                           │
 │                 │  Reranker     │   RETRIEVAL                              │
 │                 │  CrossEncoder │                                           │
 │                 │  or LLM-based │                                           │
 │ ─ ─ ─ ─ ─ ─ ─ └───────┬───────┘─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
 │                         │                                                   │
 │                         ▼                                                   │
 │                 ┌───────────────┐   ┌───────────────┐                      │
 │                 │  RAG Chain    │──▶│ Citation      │                      │
 │                 │  Context +    │   │ Parser &      │                      │
 │                 │  Generation   │   │ Validator     │                      │
 │                 └───────┬───────┘   └───────────────┘                      │
 │                         │                                                   │
 │                         ▼                       GENERATION                  │
 │                 ┌───────────────┐                                           │
 │                 │  FallbackLLM  │                                           │
 │                 │ Claude ──▶ GPT│                                           │
 │                 └───────┬───────┘                                           │
 │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
 │                         ▼                                                   │
 │                 ┌───────────────┐                                           │
 │                 │  RAGResponse  │                                           │
 │                 │  answer +     │                                           │
 │                 │  sources +    │                                           │
 │                 │  citations +  │                                           │
 │                 │  metadata     │                                           │
 │                 └───────────────┘                                           │
 └─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Document Ingestion
- **Multi-format loading** -- PDF (PyPDF2), Markdown, plain text, and web pages (httpx + BeautifulSoup)
- **Smart chunking** -- Recursive character splitting and semantic chunking (cosine similarity grouping)
- **Preprocessing pipeline** -- Unicode normalization, whitespace cleanup, metadata extraction, SHA-256 deduplication
- **Factory pattern** -- `get_loader("file.pdf")` auto-detects the right loader

### Embeddings & Storage
- **OpenAI embeddings** -- text-embedding-3-small with configurable dimensions, automatic token truncation
- **File-based embedding cache** -- SHA-256 keyed, sharded directories, TTL support, batch-aware (only uncached texts hit the API)
- **ChromaDB vector store** -- Persistent storage, configurable distance metrics (cosine/L2/IP), metadata filtering, batch upsert

### Hybrid Retrieval
- **Semantic search** -- Dense vector retrieval through the embedder + vector store
- **BM25 keyword search** -- Okapi BM25 sparse retrieval with normalized scoring
- **Reciprocal Rank Fusion** -- Combines both retrievers with configurable weights (default 0.7 semantic / 0.3 keyword)
- **Query expansion** -- LLM generates alternative phrasings to improve recall
- **HyDE** -- Hypothetical Document Embedding for better query-document alignment

### Reranking
- **Cross-encoder reranker** -- ms-marco-MiniLM-L-6-v2 scores each (query, document) pair directly. Lazy model loading.
- **LLM reranker** -- Claude/GPT scores relevance 1-10 with batch processing. Handles parse failures gracefully.

### RAG Generation
- **Dual-LLM support** -- Claude 3.5 Sonnet (primary) and GPT-4o (secondary), both with tenacity retry (3 attempts, exponential backoff)
- **Automatic fallback** -- `FallbackLLM` tries Claude first, switches to GPT-4o on failure, tracks fallback frequency
- **Citation validation** -- Parses `[Source: filename, chunk N]` citations, strips hallucinated sources
- **Graceful degradation** -- Each pipeline stage has error handling: retrieval failure, reranker failure, and LLM failure all produce meaningful responses
- **Token tracking** -- Cumulative input/output token counts across all LLM calls

### Production-Ready
- **FastAPI** with Pydantic schemas, health checks, and OpenAPI docs
- **Structured logging** via structlog (JSON in production, console in dev)
- **Prometheus metrics** definitions for monitoring
- **Docker** multi-stage build + docker-compose (API + ChromaDB + Streamlit UI)
- **GitHub Actions CI** -- lint, type-check, test on every push/PR
- **166 passing tests** covering unit, integration, and edge cases

## Quick Start

### Prerequisites

- Python 3.11+
- API keys: [Anthropic](https://console.anthropic.com/) and/or [OpenAI](https://platform.openai.com/)

### Install

```bash
git clone https://github.com/dennisdarko/rag-pipeline-production.git
cd rag-pipeline-production

# Create virtual environment and install
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Seed the Database

```bash
make seed
```

This loads the 4 sample technical documents, chunks them, generates embeddings via OpenAI, and stores everything in ChromaDB.

### Run

```bash
make run
```

API available at `http://localhost:8000` -- interactive docs at `http://localhost:8000/docs`.

### Docker

```bash
make docker-build
make docker-up
```

Starts the API server (port 8000), ChromaDB (port 8001), and Streamlit UI (port 8501).

## Usage

### Python API

```python
from src.embeddings.embedder import OpenAIEmbedder
from src.generation.chain import RAGChain
from src.generation.llm import LLMFactory
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever, SemanticRetriever, BM25Retriever
from src.vectorstore.chroma_store import ChromaVectorStore

# Set up components
embedder = OpenAIEmbedder()
store = ChromaVectorStore(collection_name="rag_documents")
semantic = SemanticRetriever(embedder=embedder, vector_store=store)
bm25 = BM25Retriever(documents=your_documents)
retriever = HybridRetriever(semantic=semantic, bm25=bm25)
reranker = CrossEncoderReranker()
llm = LLMFactory.create("fallback")

# Build the chain
chain = RAGChain(retriever=retriever, llm=llm, reranker=reranker)

# Query
response = chain.query("How do transformer attention mechanisms work?")

print(response.answer)
# "Transformer attention mechanisms use scaled dot-product attention..."

print(response.sources)
# [Source(source_name='transformer_architecture.md', chunk_index=2, ...)]

print(response.metadata)
# {'latency_ms': 1234.5, 'num_retrieved': 10, 'tokens_used': {...}, ...}
```

### REST API

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?", "top_k": 5}'

# Ingest new documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "./data/my_docs/", "doc_type": "markdown"}'

# Health check
curl http://localhost:8000/health
```

## Project Structure

```
rag-pipeline-production/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py             #   App entry point, /health endpoint
│   │   ├── schemas.py          #   Pydantic request/response models
│   │   ├── routes/             #   API route modules
│   │   └── middleware/         #   Rate limiting, auth
│   ├── config/
│   │   └── settings.py         # Pydantic Settings (all config from .env)
│   ├── ingestion/
│   │   ├── loader.py           # PDF, Markdown, Text, Web loaders
│   │   ├── chunker.py          # Recursive + Semantic chunking
│   │   └── preprocessor.py     # Clean, extract metadata, dedup
│   ├── embeddings/
│   │   ├── embedder.py         # BaseEmbedder + OpenAIEmbedder
│   │   └── cache.py            # File-based embedding cache + CachedEmbedder
│   ├── vectorstore/
│   │   ├── base.py             # VectorStore ABC + SearchResult
│   │   └── chroma_store.py     # ChromaDB implementation
│   ├── retrieval/
│   │   ├── retriever.py        # Semantic, BM25, Hybrid (RRF) retrievers
│   │   ├── query_transform.py  # QueryExpander, HyDE, MultiQueryRetriever
│   │   └── reranker.py         # CrossEncoder + LLM rerankers
│   ├── generation/
│   │   ├── llm.py              # Claude, OpenAI, Fallback LLM + Factory
│   │   ├── prompts.py          # RAG prompt templates + formatters
│   │   ├── chain.py            # RAGChain orchestrator + RAGResponse
│   │   └── response_parser.py  # Citation parsing + validation
│   ├── models/
│   │   └── document.py         # Universal Document model (Pydantic)
│   └── utils/
│       ├── logger.py           # Structlog configuration
│       └── monitoring.py       # Prometheus metrics
├── tests/
│   ├── unit/                   # 162 unit tests
│   │   ├── test_loader.py      #   Document loaders (13 tests)
│   │   ├── test_chunker.py     #   Chunking strategies (21 tests)
│   │   ├── test_embedder.py    #   OpenAI embedder (10 tests)
│   │   ├── test_cache.py       #   Embedding cache (17 tests)
│   │   ├── test_retriever.py   #   Retrieval strategies (20 tests)
│   │   ├── test_reranker.py    #   Reranking (18 tests)
│   │   ├── test_llm.py         #   LLM abstraction (27 tests)
│   │   ├── test_prompts.py     #   Prompt templates (17 tests)
│   │   ├── test_chain.py       #   RAG chain (14 tests)
│   │   ├── test_settings.py    #   Configuration (2 tests)
│   │   └── test_api_health.py  #   Health endpoint (1 test)
│   └── integration/            # 4 integration tests
│       └── test_ingestion_pipeline.py
├── data/
│   └── sample_docs/            # 4 technical articles (AI agents, MLOps, etc.)
├── scripts/
│   ├── setup.sh                # Environment setup
│   └── seed_db.sh              # Load → chunk → embed → store pipeline
├── docker/
│   ├── Dockerfile              # Multi-stage production build
│   └── docker-compose.yml      # API + ChromaDB + Streamlit UI
├── docs/
│   ├── ARCHITECTURE.md         # System design and data flow
│   ├── DEPLOYMENT.md           # Deployment guide
│   ├── EVALUATION.md           # Evaluation methodology
│   └── HANDOFF.md              # Development handoff document
├── .github/workflows/ci.yml    # GitHub Actions (lint, type-check, test)
├── pyproject.toml              # Dependencies, build config, tool settings
├── Makefile                    # Development commands
└── .env.example                # Environment variable template
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | Async HTTP server with OpenAPI docs |
| **Vector DB** | ChromaDB | Persistent vector storage with metadata filtering |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim dense vectors, tiktoken token counting |
| **LLM (Primary)** | Claude 3.5 Sonnet | High-quality generation with citation support |
| **LLM (Fallback)** | GPT-4o | Automatic fallback on Claude failures |
| **Sparse Retrieval** | rank-bm25 | Okapi BM25 keyword matching |
| **Reranking** | sentence-transformers | Cross-encoder ms-marco-MiniLM-L-6-v2 |
| **Chunking** | LangChain text splitters | Recursive character + semantic chunking |
| **Config** | Pydantic Settings | Type-safe configuration from .env |
| **Logging** | structlog | Structured JSON logging with context |
| **Retry** | tenacity | Exponential backoff for API calls |
| **Testing** | pytest + pytest-asyncio | 166 tests, async support, coverage |
| **Linting** | ruff + mypy | Fast linting + strict type checking |
| **CI/CD** | GitHub Actions | Lint, type-check, test on push/PR |
| **Containers** | Docker + Compose | Multi-stage build, 3-service stack |
| **UI** | Streamlit | Interactive query dashboard |

## Testing

```bash
# Run all 166 tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/unit/test_chain.py -v

# Run only integration tests
pytest tests/integration/ -v
```

**Test breakdown:**
- Unit tests: 162 (loaders, chunkers, embedder, cache, retrievers, rerankers, LLM, prompts, chain)
- Integration tests: 4 (full ingestion pipeline with mocked embeddings)

All external APIs (OpenAI, Anthropic, ChromaDB) are mocked in tests -- no API keys required to run the test suite.

## CI/CD

GitHub Actions runs on every push and PR to `main`:

1. **Lint** -- `ruff check` for code quality
2. **Format** -- `ruff format --check` for consistent style
3. **Type check** -- `mypy` with strict settings
4. **Test** -- `pytest` with coverage upload to Codecov

## Configuration

All settings are managed through environment variables (`.env` file), loaded via Pydantic Settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | | Anthropic API key for Claude |
| `OPENAI_API_KEY` | | OpenAI API key for embeddings + GPT-4o |
| `LLM_MODEL` | `claude-3-5-sonnet-20241022` | Primary LLM model |
| `LLM_FALLBACK_MODEL` | `gpt-4o` | Fallback LLM model |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `LLM_MAX_TOKENS` | `2048` | Max output tokens |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIMENSION` | `1536` | Embedding vector dimensions |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `RETRIEVAL_TOP_K` | `10` | Documents to retrieve |
| `RERANK_TOP_K` | `5` | Documents after reranking |
| `LOG_LEVEL` | `INFO` | Logging level |

## License

MIT
