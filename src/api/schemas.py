"""Pydantic request/response schemas for the RAG API."""

from pydantic import BaseModel, Field

# --- Query ---


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer")
    k: int = Field(default=10, ge=1, le=50, description="Number of documents to retrieve")
    rerank: bool = Field(default=True, description="Whether to rerank results")
    rerank_top_k: int = Field(
        default=5, ge=1, le=20, description="Documents to keep after reranking"
    )
    provider: str = Field(
        default="fallback",
        description="LLM provider: claude, openai, or fallback",
        pattern="^(claude|openai|fallback)$",
    )


class SourceSchema(BaseModel):
    """A source document used in the RAG response."""

    source_name: str
    chunk_text: str
    chunk_index: int
    relevance_score: float


class CitationSchema(BaseModel):
    """A validated citation in the response."""

    source: str
    chunk_index: int


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    answer: str
    sources: list[SourceSchema]
    citations: list[CitationSchema] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


# --- Ingest ---


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint (file path)."""

    source_path: str = Field(..., min_length=1, description="Path or URL to a document to ingest")
    doc_type: str = Field(
        default="auto",
        description="Document type: pdf, markdown, text, web, auto",
        pattern="^(pdf|markdown|text|web|auto)$",
    )


class IngestURLRequest(BaseModel):
    """Request body for ingesting from a URL."""

    url: str = Field(..., min_length=1, description="URL to scrape and ingest")


class IngestBatchRequest(BaseModel):
    """Request body for batch ingestion."""

    source_paths: list[str] = Field(
        ..., min_length=1, max_length=20, description="List of file paths or URLs"
    )
    doc_type: str = Field(default="auto", pattern="^(pdf|markdown|text|web|auto)$")


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint."""

    documents_processed: int
    chunks_created: int
    message: str


# --- Evaluate ---


class QAPair(BaseModel):
    """A question-answer pair for evaluation."""

    question: str = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)


class EvalRequest(BaseModel):
    """Request body for the /evaluate endpoint."""

    qa_pairs: list[QAPair] = Field(..., min_length=1, max_length=50)
    k: int = Field(default=10, ge=1, le=50)
    rerank: bool = Field(default=True)
    provider: str = Field(default="fallback", pattern="^(claude|openai|fallback)$")


class EvalMetrics(BaseModel):
    """Evaluation metrics for a single Q&A pair."""

    question: str
    answer: str
    ground_truth: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    latency_ms: float
    num_sources: int


class EvalResult(BaseModel):
    """Aggregate evaluation results."""

    total_questions: int
    avg_latency_ms: float
    avg_faithfulness: float | None = None
    avg_answer_relevancy: float | None = None
    avg_context_precision: float | None = None
    results: list[EvalMetrics]


class EvalResponse(BaseModel):
    """Response from the /evaluate endpoint."""

    eval_id: str
    result: EvalResult


# --- Health ---


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: str
    details: str = ""


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    version: str
    components: dict[str, ComponentHealth] = Field(default_factory=dict)


# --- Errors ---


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
    error_code: str = "INTERNAL_ERROR"
