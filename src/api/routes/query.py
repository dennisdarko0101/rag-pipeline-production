"""Query endpoint: run the full RAG pipeline."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    CitationSchema,
    ErrorResponse,
    QueryRequest,
    QueryResponse,
    SourceSchema,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Query"])


def _build_rag_chain(provider: str, rerank: bool):  # type: ignore[no-untyped-def]
    """Lazily build a RAGChain with the requested provider and reranker.

    Constructed per-request so that each request can choose its provider.
    Heavy objects (embedder, vector store) are cheap to instantiate because
    ChromaDB uses PersistentClient (opens existing files) and the embedder
    is just an API wrapper.
    """
    from src.embeddings.embedder import OpenAIEmbedder
    from src.generation.llm import LLMFactory
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.retriever import BM25Retriever, HybridRetriever, SemanticRetriever
    from src.vectorstore.chroma_store import ChromaVectorStore

    embedder = OpenAIEmbedder()
    store = ChromaVectorStore()
    semantic = SemanticRetriever(embedder=embedder, vector_store=store)
    bm25 = BM25Retriever()
    retriever = HybridRetriever(semantic=semantic, bm25=bm25)
    llm = LLMFactory.create(provider)
    reranker = CrossEncoderReranker() if rerank else None

    from src.generation.chain import RAGChain

    return RAGChain(retriever=retriever, llm=llm, reranker=reranker)


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
    },
    summary="Ask a question",
    description="Run the full RAG pipeline: retrieve → rerank → generate → parse citations.",
)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info(
        "api_query_start",
        question=request.question[:80],
        k=request.k,
        rerank=request.rerank,
        provider=request.provider,
    )

    try:
        chain = _build_rag_chain(request.provider, request.rerank)
        rag_response = chain.query(
            question=request.question,
            k=request.k,
            rerank_top_k=request.rerank_top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("api_query_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail="An internal error occurred while processing your query."
        ) from e

    sources = [
        SourceSchema(
            source_name=s.source_name,
            chunk_text=s.chunk_text,
            chunk_index=s.chunk_index,
            relevance_score=s.relevance_score,
        )
        for s in rag_response.sources
    ]

    citations = [
        CitationSchema(source=c.source, chunk_index=c.chunk_index) for c in rag_response.citations
    ]

    return QueryResponse(
        answer=rag_response.answer,
        sources=sources,
        citations=citations,
        metadata=rag_response.metadata,
    )
