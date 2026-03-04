# Handoff Document

## Current Status

**Phase:** 4 - Production Readiness (COMPLETE)
**Steps:** 1-12 complete (all phases done)
**Tests:** 272/272 passing (256 unit + 16 integration)
**Lint:** Clean (ruff check + ruff format + mypy)

## Completed Phases

### Phase 1: Foundation (Steps 1-3)

**Step 1 - Project Scaffolding**
- Project structure, `pyproject.toml`, Pydantic Settings, CI workflow
- FastAPI skeleton with `/health`, structlog logging, Prometheus metrics
- Docker multi-stage build, Makefile, README

**Step 2 - Document Ingestion Pipeline**
- Document model (`src/models/document.py`): Pydantic `Document` with metadata, fingerprinting
- Loaders: PDF, Markdown, Text, Web + `get_loader()` factory
- Chunkers: Recursive + Semantic + `create_chunker()` factory
- Preprocessor: clean, extract metadata, SHA-256 dedup, pipeline pattern
- 4 sample technical documents in `data/sample_docs/`

**Step 3 - Embeddings & Vector Store**
- OpenAI embedder with retry, batching, token counting, async
- File-based embedding cache (SHA-256 keys, sharded dirs, TTL)
- ChromaDB vector store with metadata filtering, batch upsert
- Seed script for sample documents

### Phase 2: Retrieval & Generation (Steps 4-6)

**Step 4 - Hybrid Retrieval**
- Semantic, BM25, and Hybrid (RRF) retrievers
- Query expansion, HyDE, MultiQueryRetriever
- Cross-encoder and LLM rerankers with lazy loading

**Steps 5-6 - LLM Generation & RAG Chain**
- Claude + OpenAI LLMs with FallbackLLM and LLMFactory
- RAG system/user prompts with citation instructions
- RAGChain orchestrator with per-stage timing and graceful degradation
- Citation parsing and validation (strips hallucinated sources)

### Phase 3: API, Evaluation & UI (Steps 7-9)

**Step 7 - FastAPI Backend**
- Versioned API (`/api/v1/`) with query, ingest, evaluate, health routes
- Rate limiting (sliding-window per-IP), request logging (correlation IDs)
- CORS, global exception handler, OpenAPI docs

**Step 8 - RAGAS Evaluation Framework**
- LLM-as-judge metrics (faithfulness, relevancy, precision, recall)
- 18-pair golden dataset across 4 categories
- EvalRunner, EvalReport (JSON + Markdown), run comparison
- CI quality gates, weekly scheduled evaluation

**Step 9 - Streamlit Demo Dashboard**
- Chat and evaluation tabs with dark theme
- Source cards, metric cards, pipeline timeline components
- httpx-only backend communication (clean UI/API boundary)
- Document ingestion, system status, configuration panel

### Phase 4: Production Readiness (Steps 10-12)

**Step 10 - Docker Production Build**
- Multi-stage Dockerfile with non-root user, health check, labels
- Separate Dockerfile.ui (lighter, no ML deps)
- Docker Compose with network isolation (backend internal, frontend external)
- Resource limits, health check dependencies, persistent volumes
- Comprehensive `.dockerignore`

**Step 11 - GitHub Actions CI/CD**
- CI: lint, type-check, test (Python 3.11 + 3.12), security scan (pip-audit), 80%+ coverage gate
- CD: Docker build + push to ghcr.io with SHA and latest tags
- Eval: weekly + manual, job summary tables, auto-issue on degradation, 90-day artifact retention

**Step 12 - Final Documentation & Polish**
- Portfolio-ready README with badges, architecture diagram, feature table
- Complete ARCHITECTURE.md with performance considerations and scaling strategy
- DEPLOYMENT.md with cloud deployment guide and monitoring section
- EVALUATION.md with test case guidelines and result interpretation
- LICENSE (MIT), CONTRIBUTING.md, updated Makefile with all targets

## Test Inventory

| Test File | Count | What It Covers |
|-----------|-------|---------------|
| `test_loader.py` | 13 | Document loaders (PDF, Markdown, Text, Web, factory) |
| `test_chunker.py` | 21 | Chunking strategies (Recursive, Semantic) |
| `test_embedder.py` | 10 | OpenAI embedder (batch, retry, tokens, dimensions) |
| `test_cache.py` | 17 | Embedding cache (hit/miss, TTL, CachedEmbedder) |
| `test_retriever.py` | 20 | Retrieval (Semantic, BM25, Hybrid/RRF) |
| `test_reranker.py` | 18 | Reranking (CrossEncoder, LLM) |
| `test_llm.py` | 27 | LLM abstraction (Claude, OpenAI, Fallback, Factory) |
| `test_prompts.py` | 17 | Prompt templates and context formatting |
| `test_chain.py` | 14 | RAG chain (full pipeline, errors, degradation) |
| `test_settings.py` | 2 | Configuration loading |
| `test_api_query.py` | 11 | Query endpoint |
| `test_api_ingest.py` | 11 | Ingest endpoints |
| `test_api_evaluate.py` | 8 | Evaluate endpoint |
| `test_api_health.py` | 4 | Health endpoint |
| `test_metrics.py` | 28 | Evaluation metrics (LLM-as-judge) |
| `test_eval_dataset.py` | 13 | Evaluation dataset (CRUD, filtering) |
| `test_eval_runner.py` | 20 | Eval runner (execution, reports, comparison) |
| `test_ingestion_pipeline.py` | 4 | Integration: load -> chunk -> embed -> store |
| `test_api.py` | 12 | Integration: API middleware, routes, CORS |
| **Total** | **272** | |

All external APIs (OpenAI, Anthropic, ChromaDB) are mocked -- no API keys required.

## Key Files

### Core Pipeline
- `src/ingestion/` -- loader.py, chunker.py, preprocessor.py
- `src/embeddings/` -- embedder.py, cache.py
- `src/vectorstore/` -- base.py, chroma_store.py
- `src/retrieval/` -- retriever.py, query_transform.py, reranker.py
- `src/generation/` -- llm.py, prompts.py, chain.py, response_parser.py
- `src/evaluation/` -- metrics.py, dataset.py, runner.py
- `src/models/document.py`, `src/config/settings.py`

### API
- `src/api/main.py`, `src/api/schemas.py`
- `src/api/routes/` -- query.py, ingest.py, evaluate.py, health.py
- `src/api/middleware/` -- rate_limit.py, logging.py

### UI
- `ui/app.py`, `ui/api_client.py`, `ui/components.py`, `ui/config.py`

### Infrastructure
- `docker/Dockerfile`, `docker/Dockerfile.ui`, `docker/docker-compose.yml`
- `.github/workflows/` -- ci.yml, cd.yml, eval.yml
- `Makefile`, `pyproject.toml`, `.dockerignore`

### Data & Scripts
- `data/sample_docs/` -- 4 technical articles
- `tests/eval/eval_dataset.json` -- 18 golden Q&A pairs
- `scripts/` -- setup.sh, seed_db.sh, run_eval.sh

## Architecture Decisions

- **Strategy + Factory patterns** throughout (loaders, chunkers, retrievers, rerankers, LLMs)
- **Decorator pattern** for embedding cache (CachedEmbedder wraps BaseEmbedder)
- **Pipeline pattern** for preprocessing (clean -> enrich -> dedup)
- **Graceful degradation** at every RAG stage (retrieval, reranking, generation)
- **Reciprocal Rank Fusion** for hybrid search (rank-based, no score normalization needed)
- **Lazy model loading** for cross-encoder (loaded on first call)
- **Citation validation** strips hallucinated sources from LLM output
- **Protocol-based interfaces** for evaluation (duck typing, not inheritance)
- **HTTP-only UI boundary** (Streamlit never imports internal modules)
- **Network isolation** in Docker (backend internal-only, frontend external)
- **Per-metric error isolation** in evaluation (one failure doesn't block others)

## Known Limitations

1. **BM25 index is rebuilt per-request** -- for production, maintain a persistent index
2. **Rate limiter is in-memory** -- for multi-instance, move to Redis
3. **ChromaDB is single-writer** -- for > 100K docs, switch to managed vector DB
4. **Embedding cache is file-based** -- for multi-instance, move to Redis
5. **No streaming** -- LLM responses are returned complete (SSE streaming is a future enhancement)
6. **No authentication** -- API is open; add JWT/OAuth for production use

## Future Improvements

- Streaming response support (Server-Sent Events)
- Async API endpoints for concurrent query handling
- Redis-backed rate limiter and embedding cache
- Persistent BM25 index (Elasticsearch/OpenSearch)
- User authentication (JWT)
- Conversation memory (multi-turn context)
- Advanced UI features (conversation export, multi-session)
- Managed vector DB (Pinecone/Qdrant) for scaling
