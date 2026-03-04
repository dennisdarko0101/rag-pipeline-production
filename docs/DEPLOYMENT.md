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

Available at `http://localhost:8501`. Requires the API server to be running (the UI calls the backend via HTTP).

The UI provides:
- **Chat tab** -- Ask questions, view answers with expandable source cards and pipeline metrics
- **Evaluation tab** -- Run the golden dataset, view aggregate metrics and per-question results
- **Sidebar** -- LLM provider selector, retrieval k slider, reranking toggle, system health status, document ingestion (file upload or URL)

#### UI Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `http://localhost:8000` | Backend API URL (set to `http://api:8000` in Docker) |
| `API_TIMEOUT` | `120` | Request timeout in seconds |

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

Two Dockerfiles are used:

- `docker/Dockerfile` -- API server (multi-stage: builder installs deps, production stage has non-root user, health check, labels)
- `docker/Dockerfile.ui` -- Streamlit UI (lighter image, only installs streamlit + httpx, no ML dependencies)

The `docker/docker-compose.yml` defines:
- **Network isolation** -- `backend` (internal, api <-> chromadb only) and `frontend` (externally accessible, ui <-> api)
- **Resource limits** -- CPU/memory limits and reservations per service
- **Health checks** -- API (httpx GET /health), ChromaDB (heartbeat), Streamlit (health endpoint)
- **Dependency ordering** -- UI waits for API healthy, API waits for ChromaDB started
- **Persistent volumes** -- `chroma_data` for vector store, `eval_results` for evaluation reports
- **Non-root user** -- Both Dockerfiles run as `appuser` (UID 1000)

### Custom Docker Build

```bash
# Build with a custom tag
docker build -f docker/Dockerfile -t rag-pipeline:custom .

# Run just the API container
docker run -p 8000:8000 --env-file .env rag-pipeline:custom

# Build just the UI
docker build -f docker/Dockerfile.ui -t rag-pipeline-ui:custom .
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

Three GitHub Actions workflows:

### CI (`.github/workflows/ci.yml`)

Runs on every push and PR to `main`:

| Job | Tool | What it checks |
|-----|------|---------------|
| Lint & Format | `ruff check` + `ruff format --check` | Code quality and style |
| Type Check | `mypy` | Static type analysis (strict mode) |
| Test | `pytest --cov --cov-fail-under=80` | 272 tests, Python 3.11 + 3.12, 80%+ coverage required |
| Security | `pip-audit` | Vulnerability scanning of dependencies |

### CD (`.github/workflows/cd.yml`)

Runs on push to `main` (after CI passes):

| Job | What it does |
|-----|-------------|
| Build & Push | Builds Docker images and pushes to `ghcr.io` |
| Tagging | Tags with git SHA and `latest` |
| Caching | Uses GitHub Actions cache for faster builds |

Images are pushed to:
- `ghcr.io/<owner>/rag-pipeline:latest` (API)
- `ghcr.io/<owner>/rag-pipeline-ui:latest` (UI)

### Evaluation (`.github/workflows/eval.yml`)

Runs weekly (Sunday midnight UTC) and on manual trigger:

| Feature | Description |
|---------|-------------|
| Quality gates | Fails if faithfulness or relevancy < 0.70 |
| Job summary | Posts Markdown results table to the workflow summary |
| Auto-issue | Creates a GitHub issue on quality degradation |
| Artifacts | Uploads JSON/Markdown reports (90-day retention) |

## Cloud Deployment

### AWS (EC2 + Docker Compose)

For a straightforward cloud deployment:

```bash
# 1. Launch an EC2 instance (t3.medium or larger)
#    - Amazon Linux 2 or Ubuntu 22.04
#    - Security group: open ports 8000, 8501

# 2. Install Docker
sudo yum install -y docker docker-compose-plugin   # Amazon Linux
sudo systemctl start docker

# 3. Clone and configure
git clone https://github.com/dennisdarko/rag-pipeline-production.git
cd rag-pipeline-production
cp .env.example .env
# Edit .env with your API keys

# 4. Build and start
make docker-build
make docker-up

# 5. Verify
curl http://localhost:8000/health
```

### AWS (ECS Fargate)

For production-grade managed deployment:

1. Push Docker images to ECR (or use ghcr.io images from CD pipeline)
2. Create ECS task definition with 3 containers (api, chromadb, ui)
3. Configure ALB target groups for port 8000 and 8501
4. Store API keys in AWS Secrets Manager, reference in task definition
5. Use EFS for ChromaDB persistence

### Key Considerations

| Concern | Recommendation |
|---------|---------------|
| **Secrets** | Use AWS Secrets Manager or SSM Parameter Store, never bake keys into images |
| **Persistence** | Mount EBS/EFS for ChromaDB data directory |
| **Scaling** | API is stateless; scale behind a load balancer. ChromaDB is single-writer. |
| **Monitoring** | Prometheus metrics are defined; connect to CloudWatch or Grafana |
| **HTTPS** | Terminate TLS at the load balancer (ALB/nginx) |
| **Domain** | Route 53 for DNS, ACM for SSL certificates |

## Monitoring and Logging

### Structured Logging

All logs are JSON-formatted via structlog:

```json
{"event": "request_complete", "method": "POST", "path": "/api/v1/query",
 "status_code": 200, "duration_ms": 2345.6, "request_id": "abc-123",
 "timestamp": "2024-01-15T10:30:00Z"}
```

Key fields:
- `request_id` -- Correlation ID (X-Request-ID header) for tracing requests across services
- `duration_ms` -- Per-request latency
- `event` -- Structured event type for filtering

### Health Checks

The `/health` endpoint returns component-level status:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "components": {
    "vectorstore": {
      "status": "healthy",
      "details": "Collection 'rag_documents': 156 documents"
    }
  }
}
```

Use this for load balancer health checks, uptime monitoring, and alerting.

### Prometheus Metrics

Metric definitions are in `src/utils/monitoring.py`. Connect your Prometheus scraper to the API server to collect:
- Request counts and latencies by endpoint
- LLM token usage
- Retrieval performance
