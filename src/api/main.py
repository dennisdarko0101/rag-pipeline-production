from fastapi import FastAPI

from src.api.schemas import HealthResponse

app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade Retrieval-Augmented Generation API",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        vectorstore_connected=False,
    )
