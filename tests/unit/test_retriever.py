"""Tests for retrieval strategies: semantic, BM25, and hybrid."""

from unittest.mock import MagicMock

import pytest

from src.models.document import Document
from src.retrieval.retriever import (
    BaseRetriever,
    BM25Retriever,
    HybridRetriever,
    SemanticRetriever,
)
from src.vectorstore.base import SearchResult


def _make_doc(doc_id: str, content: str) -> Document:
    return Document(doc_id=doc_id, content=content, metadata={"source": "test"})


def _make_result(doc_id: str, content: str, score: float, rank: int) -> SearchResult:
    return SearchResult(document=_make_doc(doc_id, content), score=score, rank=rank)


class TestBaseRetriever:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseRetriever()  # type: ignore[abstract]


class TestSemanticRetriever:
    def test_retrieve_calls_embedder_and_store(self) -> None:
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384

        mock_store = MagicMock()
        expected = [_make_result("d1", "doc one", 0.9, 0)]
        mock_store.search.return_value = expected

        retriever = SemanticRetriever(embedder=mock_embedder, vector_store=mock_store)
        results = retriever.retrieve("test query", k=5)

        mock_embedder.embed_text.assert_called_once_with("test query")
        mock_store.search.assert_called_once()
        assert results == expected

    def test_retrieve_passes_where_filter(self) -> None:
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1] * 384
        mock_store = MagicMock()
        mock_store.search.return_value = []

        retriever = SemanticRetriever(embedder=mock_embedder, vector_store=mock_store)
        retriever.retrieve("test", k=3, where={"source": "a.md"})

        call_kwargs = mock_store.search.call_args
        assert call_kwargs.kwargs.get("where") == {"source": "a.md"} or call_kwargs[1].get(
            "where"
        ) == {"source": "a.md"}

    def test_retrieve_empty_results(self) -> None:
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.0] * 384
        mock_store = MagicMock()
        mock_store.search.return_value = []

        retriever = SemanticRetriever(embedder=mock_embedder, vector_store=mock_store)
        results = retriever.retrieve("nothing matches")
        assert results == []


class TestBM25Retriever:
    def test_empty_index_returns_empty(self) -> None:
        retriever = BM25Retriever()
        results = retriever.retrieve("test query")
        assert results == []

    def test_index_and_retrieve(self) -> None:
        docs = [
            _make_doc("d1", "machine learning algorithms"),
            _make_doc("d2", "deep learning neural networks"),
            _make_doc("d3", "cooking recipes for pasta"),
        ]
        retriever = BM25Retriever(documents=docs)
        results = retriever.retrieve("machine learning", k=2)

        assert len(results) <= 2
        assert results[0].document.doc_id == "d1"
        assert results[0].score > 0
        assert results[0].rank == 0

    def test_scores_normalized_to_0_1(self) -> None:
        docs = [
            _make_doc("d1", "python programming language"),
            _make_doc("d2", "java programming language"),
        ]
        retriever = BM25Retriever(documents=docs)
        results = retriever.retrieve("python programming")

        for result in results:
            assert 0 <= result.score <= 1.0

    def test_top_result_has_score_1(self) -> None:
        docs = [
            _make_doc("d1", "python programming language guide"),
            _make_doc("d2", "java coding tutorials online"),
            _make_doc("d3", "cooking recipes for pasta dishes"),
        ]
        retriever = BM25Retriever(documents=docs)
        results = retriever.retrieve("python programming")

        # Top result should have normalized score of 1.0
        assert len(results) > 0
        assert results[0].score == 1.0

    def test_index_after_init(self) -> None:
        retriever = BM25Retriever()
        assert retriever.retrieve("test") == []

        docs = [
            _make_doc("d1", "hello world greetings"),
            _make_doc("d2", "goodbye farewell ending"),
            _make_doc("d3", "unrelated content here"),
        ]
        retriever.index(docs)
        results = retriever.retrieve("hello")
        assert len(results) >= 1
        assert results[0].document.doc_id == "d1"

    def test_k_limits_results(self) -> None:
        docs = [_make_doc(f"d{i}", f"document number {i}") for i in range(10)]
        retriever = BM25Retriever(documents=docs)
        results = retriever.retrieve("document number", k=3)
        assert len(results) <= 3

    def test_zero_score_docs_excluded(self) -> None:
        docs = [
            _make_doc("d1", "cat dog"),
            _make_doc("d2", "fish bird"),
        ]
        retriever = BM25Retriever(documents=docs)
        results = retriever.retrieve("xyz totally unrelated gibberish")
        # All docs should have score 0 and be excluded
        assert len(results) == 0

    def test_ranks_are_sequential(self) -> None:
        docs = [
            _make_doc("d1", "machine learning algorithms"),
            _make_doc("d2", "deep learning algorithms"),
            _make_doc("d3", "learning algorithms theory"),
        ]
        retriever = BM25Retriever(documents=docs)
        results = retriever.retrieve("learning algorithms")
        for i, result in enumerate(results):
            assert result.rank == i


class TestHybridRetriever:
    def _make_hybrid(
        self,
        semantic_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> HybridRetriever:
        semantic = MagicMock(spec=SemanticRetriever)
        semantic.retrieve.return_value = semantic_results
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.retrieve.return_value = bm25_results
        return HybridRetriever(semantic=semantic, bm25=bm25)

    def test_fuses_results_from_both_retrievers(self) -> None:
        sem = [_make_result("d1", "doc one", 0.9, 0)]
        bm25 = [_make_result("d2", "doc two", 0.8, 0)]

        hybrid = self._make_hybrid(sem, bm25)
        results = hybrid.retrieve("test query", k=5)

        # Both docs should appear
        result_ids = {r.document.doc_id for r in results}
        assert "d1" in result_ids
        assert "d2" in result_ids

    def test_deduplicates_by_doc_id(self) -> None:
        sem = [_make_result("d1", "doc one", 0.9, 0)]
        bm25 = [_make_result("d1", "doc one", 0.8, 0)]

        hybrid = self._make_hybrid(sem, bm25)
        results = hybrid.retrieve("test query", k=5)

        assert len(results) == 1
        assert results[0].document.doc_id == "d1"

    def test_respects_k_limit(self) -> None:
        sem = [_make_result(f"s{i}", f"sem {i}", 0.9 - i * 0.1, i) for i in range(5)]
        bm25 = [_make_result(f"b{i}", f"bm25 {i}", 0.8 - i * 0.1, i) for i in range(5)]

        hybrid = self._make_hybrid(sem, bm25)
        results = hybrid.retrieve("test", k=3)
        assert len(results) == 3

    def test_semantic_weight_higher_by_default(self) -> None:
        # Same rank 0 in both — semantic has weight 0.7 vs bm25 weight 0.3
        sem = [_make_result("sem_only", "semantic doc", 0.9, 0)]
        bm25 = [_make_result("bm25_only", "bm25 doc", 0.8, 0)]

        hybrid = self._make_hybrid(sem, bm25)
        results = hybrid.retrieve("test", k=2)

        # Semantic-only doc should rank higher due to higher weight
        assert results[0].document.doc_id == "sem_only"

    def test_custom_weights(self) -> None:
        sem_results = [_make_result("sem_only", "semantic doc", 0.9, 0)]
        bm25_results = [_make_result("bm25_only", "bm25 doc", 0.8, 0)]

        semantic = MagicMock(spec=SemanticRetriever)
        semantic.retrieve.return_value = sem_results
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.retrieve.return_value = bm25_results

        # Flip weights: keyword-heavy
        hybrid = HybridRetriever(
            semantic=semantic,
            bm25=bm25,
            semantic_weight=0.3,
            keyword_weight=0.7,
        )
        results = hybrid.retrieve("test", k=2)

        # BM25-only doc should rank higher now
        assert results[0].document.doc_id == "bm25_only"

    def test_empty_results(self) -> None:
        hybrid = self._make_hybrid([], [])
        results = hybrid.retrieve("test")
        assert results == []

    def test_ranks_are_sequential(self) -> None:
        sem = [_make_result(f"d{i}", f"doc {i}", 0.9 - i * 0.1, i) for i in range(3)]
        bm25 = [_make_result(f"d{i}", f"doc {i}", 0.8 - i * 0.1, i) for i in range(3)]

        hybrid = self._make_hybrid(sem, bm25)
        results = hybrid.retrieve("test", k=5)

        for i, result in enumerate(results):
            assert result.rank == i

    def test_fetches_extra_candidates(self) -> None:
        semantic = MagicMock(spec=SemanticRetriever)
        semantic.retrieve.return_value = []
        bm25 = MagicMock(spec=BM25Retriever)
        bm25.retrieve.return_value = []

        hybrid = HybridRetriever(semantic=semantic, bm25=bm25)
        hybrid.retrieve("test", k=10)

        # Should fetch k*3 = 30 candidates from each
        semantic.retrieve.assert_called_once_with("test", k=30)
        bm25.retrieve.assert_called_once_with("test", k=30)
