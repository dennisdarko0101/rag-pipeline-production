.PHONY: install dev-install test test-cov lint format type-check \
       run run-ui docker-build docker-up docker-down docker-logs \
       seed eval clean help

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:                          ## Install production dependencies
	pip install -e .

dev-install:                      ## Install with dev dependencies
	pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

lint:                             ## Run linting (ruff + mypy)
	ruff check src/ tests/ ui/
	mypy src/

format:                           ## Auto-fix lint issues and format code
	ruff check --fix src/ tests/ ui/
	ruff format src/ tests/ ui/

type-check:                       ## Run type checking only
	mypy src/

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:                             ## Run all tests
	pytest tests/ -v --tb=short

test-cov:                         ## Run tests with HTML coverage report
	pytest tests/ -v --tb=short --cov=src --cov-report=html --cov-fail-under=80

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run:                              ## Start FastAPI dev server (auto-reload)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:                           ## Start Streamlit dashboard
	streamlit run ui/app.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:                     ## Build all Docker images
	docker build -f docker/Dockerfile -t rag-pipeline .
	docker build -f docker/Dockerfile.ui -t rag-pipeline-ui .

docker-up:                        ## Start all services (API + ChromaDB + UI)
	docker compose -f docker/docker-compose.yml up -d

docker-down:                      ## Stop all services
	docker compose -f docker/docker-compose.yml down

docker-logs:                      ## Tail logs from all services
	docker compose -f docker/docker-compose.yml logs -f

# ---------------------------------------------------------------------------
# Data & Evaluation
# ---------------------------------------------------------------------------

seed:                             ## Seed ChromaDB with sample documents
	bash scripts/seed_db.sh

eval:                             ## Run RAG evaluation against golden dataset
	bash scripts/run_eval.sh

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:                            ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf eval_results/

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:                             ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
