# Handoff Document

## Current Status

**Phase:** 3 - API & Evaluation
**Step:** 8 - RAGAS Evaluation Framework (COMPLETE)
**Tests:** 272/272 passing (256 unit + 16 integration)
**Lint:** Clean (ruff + mypy)

## What's Been Done

### Phase 1, Step 1 - Project Scaffolding
- Project structure created with all directories and `__init__.py` files
- `pyproject.toml` with all dependencies configured
- Pydantic Settings (`src/config/settings.py`) loading from `.env`
- GitHub Actions CI workflow (lint, type-check, test)
- FastAPI skeleton with `/health` endpoint
- Structured logging with structlog
- Prometheus metrics definitions
- Pydantic request/response schemas
- Docker multi-stage build + docker-compose
- Makefile with common targets
- README with setup instructions

### Phase 1, Step 2 - Document Ingestion Pipeline
- **Document model** (`src/models/document.py`): Pydantic `Document` with content, metadata, doc_id, created_at, and convenience properties (source, char_count, fingerprint)
- **Document loaders** (`src/ingestion/loader.py`):
  - `PDFLoader` - Extracts text per page via PyPDF2
  - `MarkdownLoader` - Parses .md files, extracts H1 title
  - `TextLoader` - Plain text loading with UTF-8 encoding
  - `WebLoader` - Scrapes pages with httpx + BeautifulSoup, strips nav/footer/script elements
  - `get_loader()` factory - Auto-detects loader from file extension or URL
  - All loaders return `List[Document]` with metadata (source, file_type, timestamp, etc.)
- **Chunking strategies** (`src/ingestion/chunker.py`):
  - `RecursiveChunker` - Wraps LangChain `RecursiveCharacterTextSplitter`, configurable via settings
  - `SemanticChunker` - Groups sentences by embedding cosine similarity with configurable threshold
  - `create_chunker()` factory - Returns chunker by strategy name
  - All chunks preserve parent metadata + add chunk_index, total_chunks, parent_doc_id
- **Text preprocessing** (`src/ingestion/preprocessor.py`):
  - `clean_text()` - Unicode normalization, whitespace collapsing, header/footer stripping
  - `extract_metadata()` - Pulls titles, headers, and dates from text
  - `generate_fingerprint()` - SHA-256 content hash for deduplication
  - `deduplicate()` - Removes duplicates by fingerprint
  - `PreprocessingPipeline` - Chains clean -> extract_metadata -> deduplicate
- **Sample documents** (`data/sample_docs/`):
  - `ai_agents.md` - AI agent architectures (ReAct, Plan-and-Execute, Multi-Agent, Reflexion)
  - `mlops_best_practices.md` - MLOps lifecycle, experiment tracking, deployment patterns
  - `transformer_architecture.md` - Transformer internals (attention, positional encoding, scaling)
  - `rag_systems.md` - RAG system design, chunking strategies, evaluation

### Phase 1, Step 3 - Embeddings & Vector Store
- **Embedding models** (`src/embeddings/embedder.py`):
  - `BaseEmbedder` - Abstract base class with `embed_text()`, `embed_batch()`, `aembed_batch()`
  - `OpenAIEmbedder` - Wraps OpenAI text-embedding-3-small with batch support
  - Configurable batch_size (default 100), automatic batch splitting
  - Retry logic with tenacity (3 attempts, exponential backoff)
  - Token counting via tiktoken, automatic text truncation to model limits
  - Async support via `openai.AsyncOpenAI`
  - Structlog logging on every embed call (text length, latency, model, dimensions)
- **Embedding cache** (`src/embeddings/cache.py`):
  - `EmbeddingCache` - File-based caching with SHA-256 keys (model+text)
  - Stored as JSON in sharded subdirectories (2-char prefix)
  - Optional TTL support for cache expiration
  - Cache stats: hits, misses, total cached, hit rate
  - `CachedEmbedder` - Wraps any `BaseEmbedder`, checks cache before API calls
  - Batch-aware caching: only uncached texts are sent to the API
- **Vector store interface** (`src/vectorstore/base.py`):
  - `VectorStore` - Abstract interface with `add_documents()`, `search()`, `delete()`, `get_stats()`
  - `SearchResult` - Dataclass with document, score, rank
- **ChromaDB implementation** (`src/vectorstore/chroma_store.py`):
  - `ChromaVectorStore` - Persistent ChromaDB with configurable distance metric (cosine, l2, ip)
  - Collection management (create, delete, list)
  - Batch upsert in chunks of 500 for large document sets
  - Metadata filtering on search (ChromaDB where clauses)
  - Metadata sanitization (ChromaDB only supports str/int/float/bool)
  - Distance-to-similarity score conversion
- **Seed script** (`scripts/seed_db.sh`):
  - Loads all sample docs from data/sample_docs/
  - Preprocesses, chunks, embeds via OpenAI, stores in ChromaDB
  - Prints stats (total docs, chunks, collection size)
- **Unit tests**:
  - `tests/unit/test_embedder.py` - Mock OpenAI API, batch splitting, retry logic, token counting, dimensions
  - `tests/unit/test_cache.py` - Cache hit/miss, TTL, stats, CachedEmbedder routing
- **Integration test**:
  - `tests/integration/test_ingestion_pipeline.py` - End-to-end: load -> chunk -> embed -> store -> search, metadata filtering

### Phase 2, Step 4 - Hybrid Retrieval
- **Retrieval strategies** (`src/retrieval/retriever.py`):
  - `BaseRetriever` - Abstract base class with `retrieve(query, k)` -> `list[SearchResult]`
  - `SemanticRetriever` - Dense vector retrieval via embedder + vector store, supports metadata filtering
  - `BM25Retriever` - Sparse keyword retrieval using rank-bm25 (Okapi BM25), whitespace tokenizer, scores normalized to 0-1
  - `HybridRetriever` - Combines semantic + BM25 using Reciprocal Rank Fusion (RRF), configurable weights (default 0.7/0.3), rrf_k=60, fetches k*3 candidates from each retriever
- **Query transformation** (`src/retrieval/query_transform.py`):
  - `_call_llm()` - Helper that tries Claude (Anthropic) first, falls back to GPT-4o (OpenAI)
  - `QueryExpander` - LLM generates alternative phrasings; returns [original] + variants
  - `HyDE` - Hypothetical Document Embedding: LLM generates a hypothetical answer, embeds that instead of raw query
  - `MultiQueryRetriever` - Expands query via QueryExpander, retrieves for each variant, deduplicates by doc_id keeping highest score
- **Reranking** (`src/retrieval/reranker.py`):
  - `BaseReranker` - Abstract base class with `rerank(query, results, top_k)` -> `list[SearchResult]`
  - `CrossEncoderReranker` - Uses sentence-transformers cross-encoder (ms-marco-MiniLM-L-6-v2), lazy model loading, scores each (query, doc) pair
  - `LLMReranker` - Uses Claude/GPT to score relevance 1-10, batch processing to minimize API calls, handles parse failures gracefully
- **Unit tests**:
  - `tests/unit/test_retriever.py` - Semantic (mock embedder/store), BM25 (index, scoring, normalization), Hybrid (fusion, dedup, weights)
  - `tests/unit/test_reranker.py` - CrossEncoder (mock model, scoring, pairs), LLM (mock LLM, parsing, batching, clamping, truncation)

### Phase 2, Steps 5-6 - LLM Generation & RAG Chain
- **LLM abstraction** (`src/generation/llm.py`):
  - `BaseLLM` - Abstract base class with `generate()`, `agenerate()`, and `usage` property
  - `ClaudeLLM` - Wraps Anthropic API with retry (tenacity, 3 attempts, exponential backoff), token tracking (input/output per call), configurable temperature/max_tokens, async support
  - `OpenAILLM` - Wraps OpenAI API with same interface, retry logic, and token tracking
  - `FallbackLLM` - Tries primary (Claude), falls back to secondary (OpenAI) on failure, tracks fallback frequency and provider usage
  - `LLMFactory` - Creates LLM instances by provider name ("claude", "openai", "fallback")
  - `TokenUsage` - Dataclass tracking cumulative input_tokens, output_tokens, total_calls
- **Prompt templates** (`src/generation/prompts.py`):
  - `RAG_SYSTEM_PROMPT` - Instructs LLM to answer only from context, cite sources, say "I don't have enough information" when context is insufficient
  - `RAG_USER_PROMPT_TEMPLATE` - Template with {context} and {question} placeholders
  - `QUERY_EXPANSION_PROMPT` / `HYDE_PROMPT` - Templates for query transformation
  - `format_context()` - Formats search results into numbered sections with source info, truncates at MAX_CONTEXT_CHARS (12000)
  - `format_rag_prompt()` / `format_query_expansion_prompt()` / `format_hyde_prompt()` - Build final prompts
- **RAG chain** (`src/generation/chain.py`):
  - `RAGChain` - Orchestrates full pipeline: retrieve -> rerank (optional) -> format context -> generate -> parse citations
  - `RAGResponse` - Dataclass with answer, sources, citations, metadata (latency_ms, tokens_used, num_retrieved, num_reranked)
  - `Source` - Dataclass with source_name, chunk_text, chunk_index, relevance_score
  - Error handling at every stage with graceful degradation
  - Configurable: skip reranking, custom system prompt, adjustable k/rerank_top_k
  - Full structured logging with per-stage timing (retrieve_ms, rerank_ms, generate_ms)
- **Response parser** (`src/generation/response_parser.py`):
  - `parse_citations()` - Extracts [Source: filename, chunk N] citations from LLM output
  - `validate_citations()` - Checks citations against actually retrieved sources
  - `strip_invalid_citations()` - Removes hallucinated citations from response
  - `process_response()` - Full pipeline: parse -> validate -> clean
- **Unit tests**:
  - `tests/unit/test_llm.py` - Claude/OpenAI generate, retry logic, token tracking, async, FallbackLLM switching, LLMFactory
  - `tests/unit/test_prompts.py` - Template placeholders, context formatting, truncation, citation instructions
  - `tests/unit/test_chain.py` - Full pipeline, no results, LLM failure, retrieval failure, reranker failure, skippable reranking, citation validation, latency tracking

### Phase 3, Step 7 - FastAPI Backend
- **API schemas** (`src/api/schemas.py`):
  - `QueryRequest` - question, k, rerank, rerank_top_k, provider (with validation)
  - `QueryResponse` - answer, sources (SourceSchema), citations (CitationSchema), metadata
  - `IngestRequest` / `IngestResponse` - source_path, doc_type with validation patterns
  - `IngestURLRequest` / `IngestBatchRequest` - URL and batch ingestion schemas
  - `EvalRequest` / `EvalResponse` - Q&A pairs, per-question and aggregate metrics
  - `HealthResponse` - status, version, per-component health checks
  - `ErrorResponse` - consistent error format (detail + error_code)
- **Middleware** (`src/api/middleware/`):
  - `RateLimitMiddleware` - Sliding-window per-IP rate limiter, configurable via settings, rate limit headers (X-RateLimit-Limit/Remaining/Window), skips /health and /docs
  - `RequestLoggingMiddleware` - Correlation IDs (X-Request-ID), per-request timing, structured logging on start/complete/failure
- **Routes** (`src/api/routes/`):
  - `POST /api/v1/query` - Full RAG pipeline: configurable provider, reranking toggle, returns sources + citations + metadata
  - `POST /api/v1/ingest` - Load, preprocess, chunk, embed, and store documents from file path or URL
  - `POST /api/v1/ingest/upload` - Direct file upload with temp file handling and background cleanup
  - `POST /api/v1/evaluate` - Run RAG against Q&A pairs, per-question metrics, aggregate latency
  - `GET /health` - Component-level health checks (vectorstore status, document count)
- **Application** (`src/api/main.py`):
  - CORS middleware (configurable origins via `CORS_ORIGINS` env var)
  - Lifecycle management (startup/shutdown logging)
  - Global exception handler (500 with consistent error format)
  - Versioned API prefix (`/api/v1/`)
  - OpenAPI docs at `/docs`, ReDoc at `/redoc`
- **Unit tests** (45 new tests):
  - `tests/unit/test_api_query.py` - 11 tests: successful query, parameter passing, validation, pipeline errors, rate limit headers
  - `tests/unit/test_api_ingest.py` - 11 tests: successful ingest, file upload, doc type handling, error cases
  - `tests/unit/test_api_evaluate.py` - 8 tests: evaluation pipeline, error handling, custom provider, validation
  - `tests/unit/test_api_health.py` - 4 tests: healthy/degraded status, version, rate limit bypass
- **Integration tests** (12 new tests):
  - `tests/integration/test_api.py` - OpenAPI schema, route registration, middleware (request IDs, rate limits), CORS preflight, global error handler

### Phase 3, Step 8 - RAGAS Evaluation Framework
- **Evaluation metrics** (`src/evaluation/metrics.py`):
  - `RAGMetrics` - LLM-as-judge scoring with 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall)
  - `MetricResult` - Per-metric result with score (0-1) and natural-language explanation
  - `QuestionMetrics` - All 4 metrics for one question, with serialization
  - `AggregateMetrics` - Batch statistics (mean, std, min, max) across questions
  - `compute_aggregate()` - Computes aggregate stats from a list of per-question results
  - Custom LLM-based prompts with structured JSON output, regex fallback parsing
  - Per-metric error isolation: if one metric fails, others still complete
- **Evaluation dataset** (`src/evaluation/dataset.py`):
  - `QAPair` - Dataclass with question, ground_truth, contexts, category, metadata
  - `EvalDataset` - Collection of Q&A pairs with load/save JSON, filter by category
  - Built-in golden dataset support (default path: `tests/eval/eval_dataset.json`)
- **Golden dataset** (`tests/eval/eval_dataset.json`):
  - 18 hand-crafted Q&A pairs across 4 categories
  - 10 straightforward (single-source, easy-medium difficulty)
  - 3 multi-chunk (cross-document reasoning)
  - 3 unanswerable (info not in corpus)
  - 2 adversarial (false premise questions)
  - All based on the 4 sample documents
- **Evaluation runner** (`src/evaluation/runner.py`):
  - `EvalRunner.run(dataset, chain)` → `EvalReport` with per-question + aggregate metrics
  - `EvalReport` - Complete report with JSON and Markdown export, per-question scores + explanations
  - `compare_reports()` - Compares two runs, categorizes changes as improvements/regressions/unchanged
  - `ComparisonResult` - Improvements, regressions, unchanged metrics with summary
  - Protocol-based interfaces (works with any RAGChain-compatible object)
- **Evaluation script** (`scripts/run_eval.sh`):
  - Configurable: --provider, --category, --k, --rerank-top-k, --dataset, --output
  - Outputs summary table to console + saves JSON/Markdown reports to eval_results/
  - Writes latest_scores.json for CI threshold checks
- **CI workflow** (`.github/workflows/eval.yml`):
  - Scheduled weekly (Sundays 2 AM UTC) + manual dispatch
  - Quality gates: faithfulness >= 0.7, answer_relevancy >= 0.7
  - Uploads evaluation artifacts (90-day retention)
- **Unit tests** (61 new tests):
  - `tests/unit/test_metrics.py` - 28 tests: LLM response parsing, clamping, metric results, question metrics, aggregate computation, RAGMetrics with mock LLM, error tolerance, prompt contents
  - `tests/unit/test_eval_dataset.py` - 13 tests: QAPair defaults, dataset CRUD, filtering, categories, serialization roundtrip, golden dataset validation
  - `tests/unit/test_eval_runner.py` - 20 tests: runner execution, chain calling, parameter passing, error handling, context fallback, auto run IDs, report serialization (JSON + Markdown), comparison (improvements, regressions, unchanged, missing metrics), dataset-runner integration

## What's Next

**Phase 3 (continued)**: Polish & Production
- Streaming response support (SSE)
- Performance optimization (async endpoints, connection pooling)
- Streamlit UI improvements

## Key Files

- `src/config/settings.py` - All configuration (chunk_size, embedding_model, chroma_persist_dir, retrieval_top_k, rerank_top_k, llm_model, cors_origins, etc.)
- `src/models/document.py` - Universal Document model
- `src/ingestion/loader.py` - Document loaders (PDF, Markdown, Text, Web)
- `src/ingestion/chunker.py` - Chunking strategies (Recursive, Semantic)
- `src/ingestion/preprocessor.py` - Text cleaning, metadata extraction, dedup
- `src/embeddings/embedder.py` - BaseEmbedder + OpenAIEmbedder
- `src/embeddings/cache.py` - EmbeddingCache + CachedEmbedder
- `src/vectorstore/base.py` - VectorStore interface + SearchResult
- `src/vectorstore/chroma_store.py` - ChromaDB implementation
- `src/retrieval/retriever.py` - SemanticRetriever, BM25Retriever, HybridRetriever
- `src/retrieval/query_transform.py` - QueryExpander, HyDE, MultiQueryRetriever
- `src/retrieval/reranker.py` - CrossEncoderReranker, LLMReranker
- `src/generation/llm.py` - BaseLLM, ClaudeLLM, OpenAILLM, FallbackLLM, LLMFactory
- `src/generation/prompts.py` - RAG prompt templates and formatters
- `src/generation/chain.py` - RAGChain orchestrator, RAGResponse, Source
- `src/generation/response_parser.py` - Citation parsing and validation
- `src/api/main.py` - FastAPI app with CORS, lifecycle, middleware, routes
- `src/api/schemas.py` - Pydantic request/response schemas (Query, Ingest, Eval, Health, Error)
- `src/api/routes/query.py` - Query endpoint (builds RAGChain per request)
- `src/api/routes/ingest.py` - Ingest endpoint (full ingestion pipeline)
- `src/api/routes/evaluate.py` - Evaluation endpoint (Q&A pair testing)
- `src/api/routes/health.py` - Health check with component status
- `src/api/middleware/rate_limit.py` - Sliding-window per-IP rate limiter
- `src/api/middleware/logging.py` - Request logging with correlation IDs
- `src/evaluation/metrics.py` - RAGMetrics (LLM-as-judge with 4 metrics)
- `src/evaluation/dataset.py` - EvalDataset, QAPair, load/save JSON
- `src/evaluation/runner.py` - EvalRunner, EvalReport, compare_reports
- `tests/eval/eval_dataset.json` - Golden dataset (18 Q&A pairs, 4 categories)
- `scripts/seed_db.sh` - Seed ChromaDB with sample docs
- `scripts/run_eval.sh` - Configurable evaluation runner script
- `.github/workflows/eval.yml` - Scheduled evaluation with quality gates
- `data/sample_docs/` - Sample technical documents for testing

## Architecture Decisions

- **Document model** is Pydantic-based for validation and serialization, shared across all pipeline stages
- **Loaders** follow Strategy pattern with abstract base class and `get_loader()` factory
- **Chunkers** are configurable via `settings.chunk_size` and `settings.chunk_overlap`
- **Preprocessing** uses a pipeline pattern (clean -> enrich -> dedup) for composability
- **Fingerprinting** uses SHA-256 on normalized text for reliable deduplication
- **Embedder** uses Decorator pattern (`CachedEmbedder` wraps any `BaseEmbedder`) for composable caching
- **Embedding cache** is file-based with sharded directories to avoid filesystem limits on large caches
- **Vector store** uses Abstract Factory pattern - swap ChromaDB for Pinecone/Qdrant by implementing `VectorStore`
- **ChromaDB metadata** is sanitized automatically (lists -> comma-separated strings, None values dropped)
- **Retrieval** uses Strategy pattern: all retrievers implement `BaseRetriever.retrieve()`, allowing easy swapping
- **Hybrid search** uses Reciprocal Rank Fusion (RRF) to combine dense (semantic) and sparse (BM25) results
- **LLM layer** uses Strategy + Factory patterns: BaseLLM interface, LLMFactory for creation, FallbackLLM for resilience
- **Rerankers** use lazy loading (cross-encoder model loaded on first call) to avoid startup cost
- **Query transforms** are composable: MultiQueryRetriever wraps any BaseRetriever + QueryExpander
- **RAG chain** is a configurable pipeline with graceful degradation at every stage (retrieval, reranking, generation)
- **Citation validation** ensures LLM doesn't hallucinate sources -- only citations matching retrieved docs are kept
- **Token tracking** is cumulative across all LLM calls, surfaced in RAGResponse metadata
- **API routes** use lazy imports to avoid importing heavy ML models at startup
- **Rate limiter** is sliding-window per-IP, skips health/docs endpoints, returns standard rate limit headers
- **Middleware order** matters: CORS -> logging -> rate limiting (outermost first)
- **Versioned API** prefix `/api/v1/` allows future API evolution without breaking clients
- **Global exception handler** ensures all unhandled errors return a consistent JSON error format
- **LLM-as-judge** evaluation uses custom prompts instead of RAGAS library directly for stability and transparency
- **Per-metric error isolation** — if one metric eval fails, others still complete (no all-or-nothing failures)
- **Protocol-based interfaces** — EvalRunner works with any object matching the RAGChain protocol (duck typing)
- **Golden dataset** is hand-crafted from sample docs with 4 categories: straightforward, multi_chunk, unanswerable, adversarial
- **Dual export** (JSON + Markdown) — JSON for CI/programmatic use, Markdown for human review
