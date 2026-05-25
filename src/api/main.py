"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes.evaluate import router as evaluate_router
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle."""
    logger.info("app_starting", host=settings.api_host, port=settings.api_port)

    # Warm the shared BM25 index from the persisted corpus so the first query
    # already has keyword retrieval available (no API key needed for this).
    try:
        from src.retrieval.bm25_index import get_bm25_retriever
        from src.vectorstore.chroma_store import ChromaVectorStore

        bm25 = get_bm25_retriever(ChromaVectorStore())
        logger.info("bm25_index_warmed", num_documents=bm25.num_documents)
    except Exception as e:  # pragma: no cover - startup resilience
        logger.warning("bm25_warm_failed", error=str(e))

    yield
    logger.info("app_shutting_down")


app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Production-grade Retrieval-Augmented Generation API. "
        "Supports hybrid retrieval (semantic + BM25), cross-encoder reranking, "
        "dual-LLM generation with fallback, and citation validation."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middleware (order matters: outermost is executed first) ---

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging (runs before rate limiter so we log all requests)
app.add_middleware(RequestLoggingMiddleware)

# Rate limiting
app.add_middleware(RateLimitMiddleware)


# --- Global exception handler ---


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error_code": "INTERNAL_ERROR"},
    )


# --- Routes ---

app.include_router(health_router)
app.include_router(query_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(evaluate_router, prefix="/api/v1")
