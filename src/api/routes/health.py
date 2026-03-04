"""Health check endpoint."""

from fastapi import APIRouter

from src.api.schemas import ComponentHealth, HealthResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the API and its components.",
)
async def health_check() -> HealthResponse:
    components: dict[str, ComponentHealth] = {}

    # Check ChromaDB
    try:
        from src.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore()
        stats = store.get_stats()
        components["vectorstore"] = ComponentHealth(
            status="healthy",
            details=f"{stats.get('total_documents', 0)} documents in collection",
        )
    except Exception as e:
        components["vectorstore"] = ComponentHealth(status="unhealthy", details=str(e))

    overall = "healthy" if all(c.status == "healthy" for c in components.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version="0.1.0",
        components=components,
    )
