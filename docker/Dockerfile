# =============================================================================
# RAG Pipeline — Production Multi-Stage Build
# =============================================================================
# Stage 1 (builder): Install all Python dependencies
# Stage 2 (production): Minimal runtime image, non-root user
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec first (layer caching — only reinstall when deps change)
COPY pyproject.toml .

# Install Python dependencies into a prefix we can copy later
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: Production
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="Dennis Darko"
LABEL org.opencontainers.image.title="rag-pipeline"
LABEL org.opencontainers.image.description="Production-grade RAG pipeline with hybrid search, reranking, and dual-LLM generation"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/dennisdarko/rag-pipeline-production"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

# Copy installed Python packages from builder
COPY --from=builder /install/lib /usr/local/lib
COPY --from=builder /install/bin /usr/local/bin

# Copy application source
COPY src/ src/
COPY tests/eval/eval_dataset.json tests/eval/eval_dataset.json

# Create directories for runtime data
RUN mkdir -p data/chroma eval_results \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Built-in health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Run with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
