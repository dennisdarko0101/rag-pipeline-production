"""Ingest endpoint: load, preprocess, chunk, embed, and store documents."""

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from src.api.schemas import ErrorResponse, IngestRequest, IngestResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Ingest"])


def _run_ingestion(
    source_path: str, doc_type: str, display_name: str | None = None
) -> tuple[int, int]:
    """Run the full ingestion pipeline for a single source.

    Args:
        source_path: Path or URL to load from.
        doc_type: Document type hint (currently informational).
        display_name: Optional name to record as the document's source. Uploads
            arrive as a temp file, so this preserves the original filename for
            citations and source cards instead of leaking the temp path.

    Returns (documents_processed, chunks_created).
    """
    from src.config.settings import settings
    from src.embeddings.embedder import OpenAIEmbedder
    from src.ingestion.chunker import RecursiveChunker
    from src.ingestion.loader import get_loader
    from src.ingestion.preprocessor import PreprocessingPipeline
    from src.vectorstore.chroma_store import ChromaVectorStore

    loader = get_loader(source_path)
    docs = loader.load(source_path)

    if display_name:
        for d in docs:
            d.metadata["source"] = display_name

    pipeline = PreprocessingPipeline()
    processed = pipeline.run(docs)

    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk(processed)

    embedder = OpenAIEmbedder()
    embeddings = embedder.embed_batch([c.content for c in chunks])

    store = ChromaVectorStore()
    store.add_documents(chunks, embeddings)

    # Keep the shared BM25 index in sync so the new chunks are immediately
    # retrievable by keyword search, not just semantic search.
    from src.retrieval.bm25_index import add_documents as bm25_add

    bm25_add(chunks)

    return len(docs), len(chunks)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid source"},
        500: {"model": ErrorResponse, "description": "Ingestion error"},
    },
    summary="Ingest a document",
    description="Load a document from a file path or URL, process it through the ingestion pipeline, and store in the vector database.",
)
async def ingest(request: IngestRequest) -> IngestResponse:
    logger.info("api_ingest_start", source=request.source_path, doc_type=request.doc_type)

    try:
        docs_processed, chunks_created = _run_ingestion(request.source_path, request.doc_type)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400, detail=f"Source not found: {request.source_path}"
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("api_ingest_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Ingestion failed.") from e

    logger.info(
        "api_ingest_complete",
        docs_processed=docs_processed,
        chunks_created=chunks_created,
    )

    return IngestResponse(
        documents_processed=docs_processed,
        chunks_created=chunks_created,
        message=f"Successfully ingested {docs_processed} document(s) into {chunks_created} chunks.",
    )


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        500: {"model": ErrorResponse, "description": "Ingestion error"},
    },
    summary="Upload and ingest a file",
    description="Upload a file directly and ingest it into the vector database.",
)
async def ingest_upload(file: UploadFile, background_tasks: BackgroundTasks) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    import tempfile
    from pathlib import Path

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    logger.info("api_ingest_upload", filename=file.filename, size_bytes=len(content))

    try:
        docs_processed, chunks_created = _run_ingestion(
            tmp_path, "auto", display_name=file.filename
        )
    except Exception as e:
        # Surface the real cause (parse/embedding error) instead of failing opaquely.
        logger.error("api_ingest_upload_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e
    finally:
        # Clean up temp file in background
        background_tasks.add_task(Path(tmp_path).unlink, missing_ok=True)

    # Fail loudly rather than reporting a successful upload that indexed nothing.
    if chunks_created == 0:
        logger.warning("api_ingest_upload_empty", filename=file.filename)
        raise HTTPException(
            status_code=400,
            detail=f"No extractable text found in '{file.filename}'. Nothing was indexed.",
        )

    return IngestResponse(
        documents_processed=docs_processed,
        chunks_created=chunks_created,
        message=f"Successfully ingested '{file.filename}' into {chunks_created} chunks.",
    )
