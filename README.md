# RAG Pipeline Production

A production-grade Retrieval-Augmented Generation (RAG) system with evaluation framework.

## Features

- **Document Ingestion**: Load and chunk PDFs, Markdown, text files, and web pages
- **Hybrid Search**: Semantic similarity + BM25 keyword matching with reciprocal rank fusion
- **Cross-Encoder Reranking**: Fine-grained relevance scoring for retrieved documents
- **LLM Generation**: Claude (primary) + GPT-4o (fallback) with citation support
- **RAGAS Evaluation**: Automated quality metrics (faithfulness, relevancy, precision, recall)
- **Production API**: FastAPI backend with rate limiting, logging, and monitoring
- **Demo UI**: Streamlit dashboard for interactive queries

## Quick Start

### Prerequisites

- Python 3.11+
- API keys for Anthropic and OpenAI

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/rag-pipeline-production.git
cd rag-pipeline-production

# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Seed the vector database with sample documents
make seed

# Run the API server
make run
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Run with Docker

```bash
make docker-build
make docker-up
```

## Usage

### Query the RAG system

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'
```

### Ingest documents

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_path": "./data/sample_docs/", "doc_type": "auto"}'
```

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Lint and type check
make lint

# Format code
make format

# Run evaluation suite
make eval
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design details.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| Vector DB | ChromaDB |
| Embeddings | OpenAI text-embedding-3-small |
| LLM | Claude 3.5 Sonnet / GPT-4o |
| Evaluation | RAGAS |
| Reranking | Cross-encoder (sentence-transformers) |
| UI | Streamlit |

## License

MIT
