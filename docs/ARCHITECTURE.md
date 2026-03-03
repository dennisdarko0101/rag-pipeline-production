# Architecture

## System Overview

The RAG pipeline processes documents and user queries through four major stages, with a Streamlit UI layer on top:

```
 UI (Streamlit)           INGESTION            STORAGE              RETRIEVAL              GENERATION
 ──────────────           ─────────            ───────              ─────────              ──────────

 ┌──────────┐
 │Streamlit │ httpx
 │ Chat tab │──────▶┌────────┐               ┌─────────┐          ┌──────────┐          ┌──────────┐
 │ Eval tab │       │ Loader │──▶ Preprocess  │ChromaDB │◀── Embed │ Semantic │──┐      │ RAGChain │
 │ Sidebar  │       │PDF/MD/ │    ──▶ Chunk   │ Vector  │          │Retriever │  │ RRF  │ Context  │
 │  Config  │       │Txt/Web │    ──▶ Embed   │ Store   │          └──────────┘  ├─────▶│ Format   │
 │  Status  │       └────────┘    ──▶ Store   │         │          ┌──────────┐  │      │ Generate │
 │  Ingest  │                                 │ (cosine │          │  BM25    │──┘      │ Parse    │
 └──────────┘                                 │  HNSW)  │          │Retriever │         └────┬─────┘
       │                                      └─────────┘          └──────────┘              │
       │  FastAPI                                                        │                   ▼
       └──▶ :8000                                                        ▼            ┌──────────┐
                                                                   ┌──────────┐       │ Fallback │
                                                                   │ Reranker │       │   LLM    │
                                                                   │CrossEnc. │       │Claude/GPT│
                                                                   │ or LLM   │       └──────────┘
                                                                   └──────────┘
```

## Data Flow

### 1. Ingestion Pipeline

```
Document File/URL
       │
       ▼
┌─────────────┐     Loader detects format from extension/URL.
│  Loader     │     PDFLoader: page-level extraction via PyPDF2
│  (factory)  │     MarkdownLoader: extracts H1 title as metadata
│             │     WebLoader: httpx fetch + BeautifulSoup (strips nav/footer/script)
└──────┬──────┘
       │ List[Document]
       ▼
┌─────────────┐     clean_text(): Unicode NFC, collapse whitespace, strip headers/footers
│ Preprocessor│     extract_metadata(): titles, headers, dates from content
│  Pipeline   │     generate_fingerprint(): SHA-256 hash of normalized text
│             │     deduplicate(): remove docs with duplicate fingerprints
└──────┬──────┘
       │ List[Document] (cleaned, enriched, deduplicated)
       ▼
┌─────────────┐     RecursiveChunker: wraps LangChain RecursiveCharacterTextSplitter
│  Chunker    │       splits on ["\n\n", "\n", ". ", " ", ""] hierarchy
│  (factory)  │     SemanticChunker: groups sentences by embedding cosine similarity
│             │     Both preserve parent metadata + add chunk_index, parent_doc_id
└──────┬──────┘
       │ List[Document] (chunks with lineage metadata)
       ▼
┌─────────────┐     OpenAIEmbedder: text-embedding-3-small, 1536 dimensions
│  Embedder   │     Batches of 100, tiktoken token counting, auto-truncation
│  + Cache    │     CachedEmbedder checks file cache before API call
│             │     Cache: SHA-256(model+text) key, sharded dirs, optional TTL
└──────┬──────┘
       │ List[List[float]] (embedding vectors)
       ▼
┌─────────────┐     ChromaVectorStore: persistent HNSW index
│  ChromaDB   │     Batch upsert in chunks of 500
│  Store      │     Metadata sanitized (lists→strings, None dropped)
│             │     Configurable distance: cosine, L2, inner product
└─────────────┘
```

### 2. Retrieval Pipeline

```
User Query: "How do transformers handle long sequences?"
       │
       ▼
┌─────────────┐     Optional: QueryExpander generates alternative phrasings
│  Query      │     Optional: HyDE generates hypothetical answer, embeds that
│  Transform  │     MultiQueryRetriever: expand → retrieve for each → deduplicate
└──────┬──────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌─────────────┐     ┌──────────────┐
│  Semantic   │     │   BM25       │
│  Retriever  │     │  Retriever   │
│             │     │              │
│ embed query │     │ tokenize +   │
│ → ChromaDB  │     │ score with   │
│   .search() │     │ Okapi BM25   │
└──────┬──────┘     └──────┬───────┘
       │ k*3 results       │ k*3 results
       └──────────┬────────┘
                  ▼
          ┌───────────────┐     Reciprocal Rank Fusion:
          │ Hybrid Fusion │       score(d) = Σ weight_i / (k + rank_i)
          │ (RRF, k=60)  │       semantic_weight=0.7, keyword_weight=0.3
          │               │       Deduplicates by doc_id, sorts by fused score
          └───────┬───────┘
                  │ top-k results
                  ▼
          ┌───────────────┐     CrossEncoderReranker: ms-marco-MiniLM-L-6-v2
          │   Reranker    │       scores each (query, doc_content) pair directly
          │   (optional)  │       more accurate than bi-encoder, but slower
          │               │     LLMReranker: Claude/GPT scores 1-10
          └───────┬───────┘       batch processing, handles parse failures
                  │ rerank_top_k results
                  ▼
```

### 3. Generation Pipeline

```
Reranked Results (top-k documents)
       │
       ▼
┌─────────────┐     format_context(): numbered sections with source info
│  Prompt     │       "[1] Source: file.md, Chunk 3\n{content}"
│  Builder    │       Truncates at 12,000 chars with [..truncated] marker
│             │     RAG_SYSTEM_PROMPT: answer only from context, cite sources
└──────┬──────┘
       │ system prompt + user prompt
       ▼
┌─────────────┐     FallbackLLM wraps two providers:
│  LLM        │       1. Try ClaudeLLM (Anthropic API)
│  (Fallback) │       2. On failure → try OpenAILLM (OpenAI API)
│             │     Both: tenacity retry (3 attempts, exponential backoff)
│             │     Both: track input_tokens + output_tokens per call
└──────┬──────┘
       │ raw LLM response with citations
       ▼
┌─────────────┐     parse_citations(): regex extracts [Source: file, chunk N]
│  Response   │     validate_citations(): check against actually-retrieved sources
│  Parser     │     strip_invalid_citations(): remove hallucinated references
│             │     Clean up whitespace artifacts from removal
└──────┬──────┘
       │
       ▼
┌─────────────┐     answer: cleaned text with valid citations only
│ RAGResponse │     sources: List[Source] with name, text, index, score
│             │     citations: List[Citation] validated against retrieved docs
│             │     metadata: latency_ms, tokens_used, retrieve_ms, rerank_ms,
│             │               generate_ms, num_retrieved, num_reranked
└─────────────┘
```

## Component Descriptions

### Ingestion (`src/ingestion/`)

| Component | File | Purpose |
|-----------|------|---------|
| `DocumentLoader` | `loader.py` | Abstract base. Implementations: `PDFLoader`, `MarkdownLoader`, `TextLoader`, `WebLoader` |
| `get_loader()` | `loader.py` | Factory that picks loader from file extension or URL |
| `BaseChunker` | `chunker.py` | Abstract base. Implementations: `RecursiveChunker`, `SemanticChunker` |
| `create_chunker()` | `chunker.py` | Factory that picks chunker by strategy name |
| `PreprocessingPipeline` | `preprocessor.py` | Chains: clean → extract_metadata → fingerprint → deduplicate |

### Embeddings (`src/embeddings/`)

| Component | File | Purpose |
|-----------|------|---------|
| `BaseEmbedder` | `embedder.py` | Abstract base with `embed_text()`, `embed_batch()`, `aembed_batch()` |
| `OpenAIEmbedder` | `embedder.py` | OpenAI API wrapper with retry, batching, token counting, truncation |
| `EmbeddingCache` | `cache.py` | File-based cache with SHA-256 keys, sharded dirs, TTL |
| `CachedEmbedder` | `cache.py` | Decorator that checks cache before calling wrapped embedder |

### Vector Store (`src/vectorstore/`)

| Component | File | Purpose |
|-----------|------|---------|
| `VectorStore` | `base.py` | Abstract interface: `add_documents()`, `search()`, `delete()`, `get_stats()` |
| `SearchResult` | `base.py` | Dataclass: `document`, `score`, `rank` |
| `ChromaVectorStore` | `chroma_store.py` | ChromaDB implementation with metadata filtering, batch upsert, distance conversion |

### Retrieval (`src/retrieval/`)

| Component | File | Purpose |
|-----------|------|---------|
| `BaseRetriever` | `retriever.py` | Abstract base with `retrieve(query, k)` |
| `SemanticRetriever` | `retriever.py` | Embeds query → searches vector store |
| `BM25Retriever` | `retriever.py` | Okapi BM25 sparse keyword retrieval |
| `HybridRetriever` | `retriever.py` | Combines Semantic + BM25 via Reciprocal Rank Fusion |
| `QueryExpander` | `query_transform.py` | LLM generates alternative query phrasings |
| `HyDE` | `query_transform.py` | LLM generates hypothetical document, embeds that instead |
| `MultiQueryRetriever` | `query_transform.py` | Expands query → retrieves for each → deduplicates by doc_id |
| `BaseReranker` | `reranker.py` | Abstract base with `rerank(query, results, top_k)` |
| `CrossEncoderReranker` | `reranker.py` | Sentence-transformers cross-encoder, lazy model loading |
| `LLMReranker` | `reranker.py` | Claude/GPT relevance scoring 1-10, batch processing |

### Generation (`src/generation/`)

| Component | File | Purpose |
|-----------|------|---------|
| `BaseLLM` | `llm.py` | Abstract base with `generate()`, `agenerate()`, `usage` property |
| `ClaudeLLM` | `llm.py` | Anthropic API with retry, token tracking, async |
| `OpenAILLM` | `llm.py` | OpenAI API with same interface |
| `FallbackLLM` | `llm.py` | Tries primary, falls back to secondary, tracks fallback rate |
| `LLMFactory` | `llm.py` | Creates LLM by provider name: "claude", "openai", "fallback" |
| `TokenUsage` | `llm.py` | Cumulative token counter: input, output, total calls |
| Prompt templates | `prompts.py` | RAG system/user prompts, query expansion, HyDE templates |
| `format_context()` | `prompts.py` | Formats search results into numbered context with truncation |
| `RAGChain` | `chain.py` | Orchestrates: retrieve → rerank → format → generate → parse |
| `RAGResponse` | `chain.py` | answer + sources + citations + metadata (timings, tokens) |
| `parse_citations()` | `response_parser.py` | Regex extraction of [Source: file, chunk N] |
| `validate_citations()` | `response_parser.py` | Checks citations against actual retrieved sources |
| `process_response()` | `response_parser.py` | Full pipeline: parse → validate → strip invalid |

### API (`src/api/`)

| Component | File | Purpose |
|-----------|------|---------|
| `app` | `main.py` | FastAPI app with CORS, lifecycle, middleware stack, route registration |
| `QueryRequest` | `schemas.py` | Request schema: question, k, rerank, rerank_top_k, provider |
| `QueryResponse` | `schemas.py` | Response schema: answer, sources, citations, metadata |
| `IngestRequest` | `schemas.py` | Request schema: source_path, doc_type |
| `EvalRequest` | `schemas.py` | Request schema: qa_pairs, k, rerank, provider |
| `HealthResponse` | `schemas.py` | Response schema: status, version, per-component health |
| `ErrorResponse` | `schemas.py` | Consistent error format: detail, error_code |
| `query` | `routes/query.py` | POST /api/v1/query -- builds RAGChain per request, returns validated response |
| `ingest` | `routes/ingest.py` | POST /api/v1/ingest -- full ingestion pipeline from path or URL |
| `ingest_upload` | `routes/ingest.py` | POST /api/v1/ingest/upload -- file upload with temp file cleanup |
| `evaluate` | `routes/evaluate.py` | POST /api/v1/evaluate -- runs RAG against Q&A pairs, returns metrics |
| `health_check` | `routes/health.py` | GET /health -- component-level health (vectorstore status) |
| `RateLimitMiddleware` | `middleware/rate_limit.py` | Sliding-window per-IP rate limiter with X-RateLimit headers |
| `RequestLoggingMiddleware` | `middleware/logging.py` | Correlation IDs (X-Request-ID), per-request timing |

### Evaluation (`src/evaluation/`)

| Component | File | Purpose |
|-----------|------|---------|
| `RAGMetrics` | `metrics.py` | LLM-as-judge scorer: faithfulness, answer_relevancy, context_precision, context_recall |
| `MetricResult` | `metrics.py` | Per-metric result with score (0-1) and explanation |
| `QuestionMetrics` | `metrics.py` | All 4 metrics for one question, serializable to dict |
| `AggregateMetrics` | `metrics.py` | Batch statistics: mean, std, min, max per metric |
| `compute_aggregate()` | `metrics.py` | Computes aggregate stats from a list of QuestionMetrics |
| `QAPair` | `dataset.py` | Dataclass: question, ground_truth, contexts, category, metadata |
| `EvalDataset` | `dataset.py` | Collection of Q&A pairs with load/save JSON, filter by category |
| `EvalRunner` | `runner.py` | Orchestrates evaluation: runs RAGChain on each Q&A pair, scores with RAGMetrics |
| `EvalReport` | `runner.py` | Complete report: per-question metrics, aggregate stats, JSON/Markdown export |
| `compare_reports()` | `runner.py` | Compares two EvalReports, categorizes changes as improvements/regressions/unchanged |

## Design Decisions

### Why ChromaDB?

ChromaDB was chosen as the vector store for several reasons:
- **Zero infrastructure** -- runs as an embedded library with persistent file storage, no separate server needed for development
- **HNSW indexing** -- efficient approximate nearest neighbor search
- **Metadata filtering** -- supports `where` clauses to filter by source, file type, etc.
- **Simple API** -- straightforward Python client, upsert semantics for idempotent ingestion

The `VectorStore` abstract interface makes it easy to swap in Pinecone, Qdrant, or Weaviate for production at scale -- just implement 4 methods.

### Why Hybrid Retrieval (Semantic + BM25)?

Neither pure semantic search nor pure keyword search is sufficient:
- **Semantic search** excels at understanding intent but can miss exact term matches (e.g., specific error codes, function names)
- **BM25 keyword search** catches exact matches but misses semantic similarity (e.g., "ML" vs "machine learning")

Combining both with **Reciprocal Rank Fusion** (RRF) captures the strengths of each. The formula `score(d) = Σ weight / (k + rank)` is robust because it uses rank positions rather than raw scores, making it insensitive to score distribution differences between retrievers.

Default weights (0.7 semantic / 0.3 keyword) favor semantic understanding while still boosting exact-match results.

### Why RRF over Other Fusion Methods?

- **Score normalization is hard** -- semantic similarity scores and BM25 scores live in different distributions. Normalizing them for weighted averaging introduces artifacts.
- **RRF is rank-based** -- only uses ordinal positions, so it's agnostic to score distributions
- **k=60 is standard** -- the constant prevents top-ranked documents from dominating. This value comes from the original RRF paper and works well empirically.
- **Simple and effective** -- no hyperparameters to tune beyond the weights

### Why Cross-Encoder Reranking?

Bi-encoder retrieval (embed query and docs separately, compare) is fast but imprecise. Cross-encoders process the (query, document) pair together through all transformer layers, enabling full attention between query tokens and document tokens. This produces much more accurate relevance scores, especially for nuanced queries.

ms-marco-MiniLM-L-6-v2 was chosen because:
- Trained specifically on passage ranking (MS MARCO)
- Small enough for CPU inference (6 layers, 22M params)
- Lazy loading avoids startup cost when reranking isn't needed

### Why Dual-LLM with Fallback?

Production systems need resilience. API outages happen. The `FallbackLLM` pattern ensures:
- If Claude's API is down, the system automatically switches to GPT-4o
- Fallback frequency is tracked for monitoring (if fallback rate spikes, investigate)
- Both providers use the same `BaseLLM` interface, so the rest of the pipeline is provider-agnostic

### Why Validate Citations?

LLMs hallucinate. Even when instructed to cite only from provided context, models sometimes invent source references. The response parser:
1. Extracts all `[Source: file, chunk N]` citations from the response
2. Checks each against the set of actually-retrieved source filenames
3. Strips any citation that doesn't match a real source

This ensures every citation in the final response points to a document that was actually retrieved and used as context.

### UI (`ui/`)

| Component | File | Purpose |
|-----------|------|---------|
| `app` | `app.py` | Streamlit dashboard: sidebar (config, status, ingestion), chat tab, evaluation tab |
| `api_client` | `api_client.py` | httpx HTTP client wrapping all FastAPI endpoints (health, query, ingest, evaluate) |
| `metric_card()` | `components.py` | Color-coded metric display (green >= 0.8, yellow >= 0.6, red < 0.6) |
| `source_card()` | `components.py` | Retrieved source with relevance score bar and truncated content |
| `pipeline_timeline()` | `components.py` | Visual timeline of pipeline stage timings (retrieve, rerank, generate) |
| `status_indicator()` | `components.py` | Green/red status dot for system health display |
| `COLORS` | `config.py` | Dark theme color palette (slate/indigo), API connection settings |
| `score_color()` | `config.py` | Maps score to color hex based on thresholds |

### Error Handling Strategy

The RAG chain uses **graceful degradation** at every stage:

| Stage | Failure | Behavior |
|-------|---------|----------|
| Retrieval | Exception | Returns error message, no LLM call |
| Retrieval | No results | Returns "I don't have enough information", no LLM call |
| Reranking | Exception | Falls back to un-reranked results (truncated to top_k) |
| Generation | Exception | Returns error message with sources still attached |
| Citation parsing | Invalid citations | Strips invalid ones, keeps valid ones |

This means the system never crashes on a user query -- it always returns a meaningful response, even if degraded.

### Why Streamlit for the UI?

Streamlit was chosen for the demo dashboard because:
- **Rapid prototyping** -- build a full interactive UI in pure Python, no frontend framework required
- **Built-in chat components** -- `st.chat_message` and `st.chat_input` provide a native chat experience
- **Session state** -- conversation history and evaluation results persist across rerenders
- **Docker-friendly** -- lightweight container, easy to add to the existing compose stack

The UI communicates with the backend **exclusively via httpx HTTP calls** (through `ui/api_client.py`), never importing internal Python modules. This enforces a clean boundary: the UI is a consumer of the REST API, just like any external client. This means the UI can be replaced with a React/Next.js frontend without touching the backend.
