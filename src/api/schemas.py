from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")


class Source(BaseModel):
    content: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    query_time_ms: float


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path or URL to document(s) to ingest")
    doc_type: str = Field(default="auto", description="Document type: pdf, markdown, text, web, auto")


class IngestResponse(BaseModel):
    documents_processed: int
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    vectorstore_connected: bool
