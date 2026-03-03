"""Tests for the RAG chain orchestrator."""

from unittest.mock import MagicMock

from src.generation.chain import RAGChain, RAGResponse, Source
from src.generation.llm import BaseLLM, TokenUsage
from src.models.document import Document
from src.retrieval.reranker import BaseReranker
from src.retrieval.retriever import BaseRetriever
from src.vectorstore.base import SearchResult


def _make_result(
    doc_id: str,
    content: str,
    source: str = "test.md",
    chunk_index: int = 0,
    score: float = 0.9,
    rank: int = 0,
) -> SearchResult:
    return SearchResult(
        document=Document(
            doc_id=doc_id,
            content=content,
            metadata={"source": source, "chunk_index": chunk_index},
        ),
        score=score,
        rank=rank,
    )


def _make_mock_retriever(results: list[SearchResult] | None = None) -> MagicMock:
    mock = MagicMock(spec=BaseRetriever)
    mock.retrieve.return_value = results or []
    return mock


def _make_mock_llm(response: str = "Test answer") -> MagicMock:
    mock = MagicMock(spec=BaseLLM)
    mock.generate.return_value = response
    mock.usage = TokenUsage()
    mock.usage.record(50, 25)
    return mock


def _make_mock_reranker(results: list[SearchResult] | None = None) -> MagicMock:
    mock = MagicMock(spec=BaseReranker)
    mock.rerank.return_value = results or []
    return mock


class TestRAGChain:
    def test_full_pipeline(self) -> None:
        results = [
            _make_result("d1", "RAG combines retrieval and generation.", source="rag.md", chunk_index=0),
            _make_result("d2", "Embeddings map text to vectors.", source="embed.md", chunk_index=1, score=0.8, rank=1),
        ]
        reranked = [results[0]]  # Reranker picks top 1

        retriever = _make_mock_retriever(results)
        reranker = _make_mock_reranker(reranked)
        llm = _make_mock_llm("RAG uses retrieval. [Source: rag.md, chunk 0]")

        chain = RAGChain(retriever=retriever, llm=llm, reranker=reranker)
        response = chain.query("What is RAG?", k=10, rerank_top_k=1)

        assert isinstance(response, RAGResponse)
        assert "RAG uses retrieval" in response.answer
        assert len(response.sources) > 0
        assert response.metadata["num_retrieved"] > 0

    def test_response_sources_match_retrieved(self) -> None:
        results = [
            _make_result("d1", "Content about RAG", source="rag.md", chunk_index=2, score=0.95),
        ]
        retriever = _make_mock_retriever(results)
        llm = _make_mock_llm("Answer [Source: rag.md, chunk 2]")

        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("question")

        assert len(response.sources) == 1
        assert response.sources[0].source_name == "rag.md"
        assert response.sources[0].chunk_index == 2
        assert response.sources[0].relevance_score == 0.95

    def test_metadata_contains_expected_fields(self) -> None:
        results = [_make_result("d1", "content")]
        retriever = _make_mock_retriever(results)
        llm = _make_mock_llm("Answer")

        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("question")

        assert "num_retrieved" in response.metadata
        assert "latency_ms" in response.metadata
        assert "tokens_used" in response.metadata
        assert "retrieve_ms" in response.metadata
        assert "generate_ms" in response.metadata
        assert isinstance(response.metadata["latency_ms"], float)

    def test_no_results_returns_default_answer(self) -> None:
        retriever = _make_mock_retriever([])
        llm = _make_mock_llm("Should not be called")

        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("obscure question")

        assert "don't have enough information" in response.answer
        assert response.metadata["num_retrieved"] == 0
        llm.generate.assert_not_called()

    def test_retrieval_failure_returns_error_response(self) -> None:
        retriever = _make_mock_retriever()
        retriever.retrieve.side_effect = RuntimeError("DB connection failed")
        llm = _make_mock_llm()

        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("question")

        assert "error" in response.answer.lower()
        assert "error" in response.metadata

    def test_llm_failure_returns_error_with_sources(self) -> None:
        results = [_make_result("d1", "content")]
        retriever = _make_mock_retriever(results)
        llm = _make_mock_llm()
        llm.generate.side_effect = RuntimeError("LLM API down")

        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("question")

        assert "error" in response.answer.lower()
        assert len(response.sources) == 1  # Sources still returned
        assert "error" in response.metadata

    def test_reranking_is_skippable(self) -> None:
        results = [
            _make_result("d1", "content", rank=0),
            _make_result("d2", "more content", rank=1),
        ]
        retriever = _make_mock_retriever(results)
        llm = _make_mock_llm("Answer from both sources")

        # No reranker
        chain = RAGChain(retriever=retriever, llm=llm, reranker=None)
        response = chain.query("question")

        assert response.metadata.get("num_reranked") == 0
        assert "rerank_ms" not in response.metadata

    def test_reranking_failure_degrades_gracefully(self) -> None:
        results = [
            _make_result("d1", "content", rank=0),
            _make_result("d2", "content 2", rank=1),
            _make_result("d3", "content 3", rank=2),
        ]
        retriever = _make_mock_retriever(results)
        reranker = _make_mock_reranker()
        reranker.rerank.side_effect = RuntimeError("Reranker model failed")
        llm = _make_mock_llm("Answer")

        chain = RAGChain(retriever=retriever, llm=llm, reranker=reranker)
        response = chain.query("question", rerank_top_k=2)

        # Should still produce an answer using un-reranked results
        assert response.answer == "Answer"
        llm.generate.assert_called_once()

    def test_invalid_citations_stripped(self) -> None:
        results = [_make_result("d1", "content", source="real.md")]
        retriever = _make_mock_retriever(results)
        llm = _make_mock_llm(
            "Answer [Source: real.md, chunk 0] and also [Source: fake.md, chunk 5]"
        )

        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("question")

        # Invalid citation should be stripped
        assert "[Source: fake.md, chunk 5]" not in response.answer
        assert "[Source: real.md, chunk 0]" in response.answer

    def test_latency_tracking(self) -> None:
        results = [_make_result("d1", "content")]
        retriever = _make_mock_retriever(results)
        llm = _make_mock_llm("Answer")

        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("question")

        assert response.metadata["latency_ms"] >= 0
        assert response.metadata["retrieve_ms"] >= 0
        assert response.metadata["generate_ms"] >= 0

    def test_custom_system_prompt(self) -> None:
        results = [_make_result("d1", "content")]
        retriever = _make_mock_retriever(results)
        llm = _make_mock_llm("Answer")

        chain = RAGChain(
            retriever=retriever,
            llm=llm,
            system_prompt="Custom system prompt",
        )
        chain.query("question")

        call_kwargs = llm.generate.call_args
        assert call_kwargs.kwargs.get("system") == "Custom system prompt" or call_kwargs[1].get("system") == "Custom system prompt"

    def test_citations_in_response(self) -> None:
        results = [
            _make_result("d1", "RAG content", source="rag.md", chunk_index=0),
            _make_result("d2", "ML content", source="ml.md", chunk_index=3, rank=1),
        ]
        retriever = _make_mock_retriever(results)
        llm = _make_mock_llm(
            "RAG is great [Source: rag.md, chunk 0]. ML too [Source: ml.md, chunk 3]."
        )

        chain = RAGChain(retriever=retriever, llm=llm)
        response = chain.query("question")

        assert len(response.citations) == 2
        assert response.citations[0].source == "rag.md"
        assert response.citations[1].source == "ml.md"


class TestRAGResponse:
    def test_default_values(self) -> None:
        response = RAGResponse(answer="Test")
        assert response.answer == "Test"
        assert response.sources == []
        assert response.citations == []
        assert response.metadata == {}


class TestSource:
    def test_source_fields(self) -> None:
        source = Source(
            source_name="doc.md",
            chunk_text="Some text content",
            chunk_index=3,
            relevance_score=0.85,
        )
        assert source.source_name == "doc.md"
        assert source.chunk_index == 3
        assert source.relevance_score == 0.85
