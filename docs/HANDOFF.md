# Handoff Document

## Current Status

**Phase:** 1 - Foundation
**Step:** 3 - Embeddings & Vector Store (COMPLETE)

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
  - `tests/integration/test_ingestion_pipeline.py` - End-to-end: load → chunk → embed → store → search, metadata filtering

## What's Next

**Phase 2, Step 1**: Retrieval pipeline
- BM25 sparse retrieval
- Hybrid search (dense + sparse)
- Reranking with cross-encoder
- Query expansion / HyDE
- Unit tests for retrieval

## Key Files

- `src/config/settings.py` - All configuration (chunk_size, embedding_model, chroma_persist_dir, etc.)
- `src/models/document.py` - Universal Document model
- `src/ingestion/loader.py` - Document loaders (PDF, Markdown, Text, Web)
- `src/ingestion/chunker.py` - Chunking strategies (Recursive, Semantic)
- `src/ingestion/preprocessor.py` - Text cleaning, metadata extraction, dedup
- `src/embeddings/embedder.py` - BaseEmbedder + OpenAIEmbedder
- `src/embeddings/cache.py` - EmbeddingCache + CachedEmbedder
- `src/vectorstore/base.py` - VectorStore interface + SearchResult
- `src/vectorstore/chroma_store.py` - ChromaDB implementation
- `src/api/main.py` - FastAPI app entry point
- `scripts/seed_db.sh` - Seed ChromaDB with sample docs
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
- **ChromaDB metadata** is sanitized automatically (lists → comma-separated strings, None values dropped)
