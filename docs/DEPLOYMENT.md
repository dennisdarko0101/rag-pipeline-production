# Deployment Guide

## Local Development

### Prerequisites

- Python 3.11+
- API keys for Anthropic and/or OpenAI

### Setup from Scratch

```bash
# 1. Clone and enter the project
git clone https://github.com/dennisdarko/rag-pipeline-production.git
cd rag-pipeline-production

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

# 3. Install with dev dependencies
pip install -e ".[dev]"

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys

# 5. Seed the vector database with sample documents
make seed

# 6. Run the API server
make run
```

Or use the setup script:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

### Running the API

```bash
# Development mode (auto-reload on code changes)
make run
# equivalent to: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API is available at:
- API: `http://localhost:8000/api/v1/`
- OpenAPI docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/query` | Run the full RAG pipeline (retrieve, rerank, generate) |
| `POST` | `/api/v1/ingest` | Ingest a document from a file path or URL |
| `POST` | `/api/v1/ingest/upload` | Upload and ingest a file directly |
| `POST` | `/api/v1/evaluate` | Evaluate RAG quality against Q&A pairs |
| `GET` | `/health` | Component-level health check |

### Running the Streamlit UI

```bash
make run-ui
# equivalent to: streamlit run ui/app.py
```

Available at `http://localhost:8501`.

### Running Tests

```bash
# All tests (no API keys required -- all external APIs are mocked)
make test

# With coverage report
make test-cov

# Specific test file
pytest tests/unit/test_chain.py -v
```

## Docker Deployment

### Architecture

The docker-compose stack runs three services:

| Service | Port | Purpose |
|---------|------|---------|
| `api` | 8000 | FastAPI backend |
| `chromadb` | 8001 | ChromaDB vector database |
| `ui` | 8501 | Streamlit query interface |

### Build and Run

```bash
# Build the Docker image
make docker-build

# Start all services
make docker-up

# Stop all services
make docker-down

# View logs
docker compose -f docker/docker-compose.yml logs -f api
```

### Docker Configuration

The `docker/Dockerfile` uses a multi-stage build:
1. **Base stage** -- installs dependencies with pip
2. **Production stage** -- copies only site-packages and source code, no build tools

The `docker/docker-compose.yml` defines:
- Shared `chroma_data` volume for ChromaDB persistence
- `.env` file mounted for API keys
- Health check dependencies between services

### Custom Docker Build

```bash
# Build with a custom tag
docker build -f docker/Dockerfile -t rag-pipeline:custom .

# Run just the API container
docker run -p 8000:8000 --env-file .env rag-pipeline:custom
```

## Environment Variables Reference

All configuration is loaded via Pydantic Settings from the `.env` file (or system environment variables).

### Required

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude LLM. Required for primary generation and LLM reranking. |
| `OPENAI_API_KEY` | OpenAI API key. Required for embeddings (text-embedding-3-small) and GPT-4o fallback. |

At minimum, you need `OPENAI_API_KEY` for embeddings. For generation, you need at least one of the two LLM keys.

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `claude-3-5-sonnet-20241022` | Primary LLM model name |
| `LLM_FALLBACK_MODEL` | `gpt-4o` | Fallback LLM model name |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature (0.0 = deterministic) |
| `LLM_MAX_TOKENS` | `2048` | Maximum output tokens per generation |

### Embedding Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIMENSION` | `1536` | Embedding vector dimensions |

### Chunking Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `512` | Target characters per chunk |
| `CHUNK_OVERLAP` | `50` | Character overlap between adjacent chunks |

### Retrieval Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | `10` | Number of documents to retrieve |
| `RERANK_TOP_K` | `5` | Number of documents to keep after reranking |

### Storage Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Directory for ChromaDB persistent storage |

### Application Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |
| `CORS_ORIGINS` | `*` | Comma-separated CORS origins |

## Seeding the Database

The seed script loads sample technical documents, processes them through the full ingestion pipeline, and stores them in ChromaDB.

### Using the Script

```bash
make seed
# or: bash scripts/seed_db.sh
```

This runs the following pipeline:
1. **Load** 4 markdown documents from `data/sample_docs/`
2. **Preprocess** (clean, extract metadata, generate fingerprints)
3. **Chunk** using RecursiveChunker (512 chars, 50 overlap)
4. **Embed** via OpenAI text-embedding-3-small
5. **Store** in ChromaDB collection `rag_documents`

Requires `OPENAI_API_KEY` to be set.

### Seeding Custom Documents

Place your documents in a directory and modify the seed script, or use the Python API directly:

```python
from src.ingestion.loader import get_loader
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.preprocessor import PreprocessingPipeline
from src.embeddings.embedder import OpenAIEmbedder
from src.vectorstore.chroma_store import ChromaVectorStore

# Load
loader = get_loader("path/to/document.pdf")
docs = loader.load("path/to/document.pdf")

# Process
pipeline = PreprocessingPipeline()
processed = pipeline.run(docs)

# Chunk
chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(processed)

# Embed and store
embedder = OpenAIEmbedder()
embeddings = embedder.embed_batch([c.content for c in chunks])
store = ChromaVectorStore()
store.add_documents(chunks, embeddings)
```

## CI/CD

GitHub Actions runs on every push and PR to `main`:

| Job | Tool | What it checks |
|-----|------|---------------|
| Lint | `ruff check` | Code quality and style rules |
| Format | `ruff format --check` | Consistent code formatting |
| Type check | `mypy` | Static type analysis (strict mode) |
| Test | `pytest --cov` | All 211 tests + coverage upload to Codecov |

Configuration: `.github/workflows/ci.yml`
