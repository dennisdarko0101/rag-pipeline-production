"""RAG chain: orchestrates retrieval, reranking, and generation."""

from dataclasses import dataclass, field
from time import perf_counter

from src.generation.llm import BaseLLM
from src.generation.prompts import RAG_SYSTEM_PROMPT, format_rag_prompt
from src.generation.response_parser import Citation, process_response
from src.retrieval.reranker import BaseReranker
from src.retrieval.retriever import BaseRetriever
from src.utils.logger import get_logger
from src.vectorstore.base import SearchResult

logger = get_logger(__name__)


@dataclass
class Source:
    """A source document used in the RAG response."""

    source_name: str
    chunk_text: str
    chunk_index: int
    relevance_score: float


@dataclass
class RAGResponse:
    """Complete response from the RAG chain."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RAGChain:
    """Orchestrates the full RAG pipeline.

    Pipeline: retrieve → rerank (optional) → format context → generate answer → parse citations.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        reranker: BaseReranker | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker
        self._system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        logger.info(
            "rag_chain_init",
            has_reranker=reranker is not None,
        )

    def query(
        self,
        question: str,
        k: int = 10,
        rerank_top_k: int = 5,
    ) -> RAGResponse:
        """Run the full RAG pipeline.

        Args:
            question: The user's question.
            k: Number of documents to retrieve.
            rerank_top_k: Number of documents after reranking.

        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        total_start = perf_counter()
        timings: dict[str, float] = {}

        # --- Stage 1: Retrieve ---
        stage_start = perf_counter()
        try:
            results = self._retriever.retrieve(question, k=k)
        except Exception as e:
            logger.error("rag_retrieve_failed", error=str(e))
            return RAGResponse(
                answer="I encountered an error while searching for relevant information.",
                metadata={
                    "error": f"retrieval_failed: {e}",
                    "latency_ms": round((perf_counter() - total_start) * 1000, 1),
                },
            )
        timings["retrieve_ms"] = round((perf_counter() - stage_start) * 1000, 1)

        if not results:
            logger.info("rag_no_results", question=question[:80])
            return RAGResponse(
                answer="I don't have enough information to answer this question.",
                metadata={
                    "num_retrieved": 0,
                    "latency_ms": round((perf_counter() - total_start) * 1000, 1),
                    **timings,
                },
            )

        # --- Stage 2: Rerank (optional) ---
        if self._reranker is not None:
            stage_start = perf_counter()
            try:
                results = self._reranker.rerank(question, results, top_k=rerank_top_k)
            except Exception as e:
                logger.warning("rag_rerank_failed", error=str(e))
                # Graceful degradation: continue with un-reranked results, truncated to top_k
                results = results[:rerank_top_k]
            timings["rerank_ms"] = round((perf_counter() - stage_start) * 1000, 1)

        # --- Stage 3: Generate ---
        stage_start = perf_counter()
        prompt = format_rag_prompt(question, results)
        try:
            raw_answer = self._llm.generate(prompt, system=self._system_prompt)
        except Exception as e:
            logger.error("rag_generate_failed", error=str(e))
            return RAGResponse(
                answer="I encountered an error while generating a response.",
                sources=self._build_sources(results),
                metadata={
                    "error": f"generation_failed: {e}",
                    "num_retrieved": len(results),
                    "latency_ms": round((perf_counter() - total_start) * 1000, 1),
                    **timings,
                },
            )
        timings["generate_ms"] = round((perf_counter() - stage_start) * 1000, 1)

        # --- Stage 4: Parse and validate citations ---
        valid_sources = {r.document.metadata.get("source", "unknown") for r in results}
        answer, citations = process_response(raw_answer, valid_sources)

        # Build sources list
        sources = self._build_sources(results)

        total_elapsed = perf_counter() - total_start
        metadata = {
            "num_retrieved": len(results),
            "num_reranked": len(results) if self._reranker is not None else 0,
            "num_citations": len(citations),
            "tokens_used": self._llm.usage.to_dict(),
            "latency_ms": round(total_elapsed * 1000, 1),
            **timings,
        }

        logger.info(
            "rag_query_complete",
            question=question[:80],
            num_results=len(results),
            num_citations=len(citations),
            latency_ms=round(total_elapsed * 1000, 1),
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            citations=citations,
            metadata=metadata,
        )

    @staticmethod
    def _build_sources(results: list[SearchResult]) -> list[Source]:
        """Convert search results to Source objects."""
        return [
            Source(
                source_name=r.document.metadata.get("source", "unknown"),
                chunk_text=r.document.content[:200],
                chunk_index=r.document.metadata.get("chunk_index", r.rank),
                relevance_score=r.score,
            )
            for r in results
        ]
