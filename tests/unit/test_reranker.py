"""Tests for reranking strategies: cross-encoder and LLM-based."""

from unittest.mock import MagicMock, patch

import pytest

from src.models.document import Document
from src.retrieval.reranker import BaseReranker, CrossEncoderReranker, LLMReranker
from src.vectorstore.base import SearchResult


def _make_doc(doc_id: str, content: str) -> Document:
    return Document(doc_id=doc_id, content=content, metadata={"source": "test"})


def _make_result(doc_id: str, content: str, score: float, rank: int) -> SearchResult:
    return SearchResult(document=_make_doc(doc_id, content), score=score, rank=rank)


class TestBaseReranker:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]


class TestCrossEncoderReranker:
    @patch("src.retrieval.reranker.CrossEncoderReranker._load_model")
    def test_rerank_sorts_by_cross_encoder_score(self, mock_load: MagicMock) -> None:
        reranker = CrossEncoderReranker()
        # Simulate model already loaded
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.3, 0.9, 0.1]

        results = [
            _make_result("d1", "doc one", 0.5, 0),
            _make_result("d2", "doc two", 0.4, 1),
            _make_result("d3", "doc three", 0.3, 2),
        ]

        reranked = reranker.rerank("test query", results, top_k=3)

        assert len(reranked) == 3
        assert reranked[0].document.doc_id == "d2"  # Highest cross-encoder score (0.9)
        assert reranked[1].document.doc_id == "d1"  # 0.3
        assert reranked[2].document.doc_id == "d3"  # 0.1

    @patch("src.retrieval.reranker.CrossEncoderReranker._load_model")
    def test_rerank_respects_top_k(self, mock_load: MagicMock) -> None:
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]

        results = [_make_result(f"d{i}", f"doc {i}", 0.5, i) for i in range(5)]
        reranked = reranker.rerank("test", results, top_k=2)

        assert len(reranked) == 2

    @patch("src.retrieval.reranker.CrossEncoderReranker._load_model")
    def test_rerank_empty_input(self, mock_load: MagicMock) -> None:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("test", [], top_k=5)
        assert reranked == []

    @patch("src.retrieval.reranker.CrossEncoderReranker._load_model")
    def test_reranked_scores_are_cross_encoder_scores(self, mock_load: MagicMock) -> None:
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.85, 0.42]

        results = [
            _make_result("d1", "doc one", 0.1, 0),
            _make_result("d2", "doc two", 0.9, 1),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        # Scores should be from cross-encoder, not original
        assert reranked[0].score == pytest.approx(0.85)
        assert reranked[1].score == pytest.approx(0.42)

    @patch("src.retrieval.reranker.CrossEncoderReranker._load_model")
    def test_ranks_are_sequential(self, mock_load: MagicMock) -> None:
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.3, 0.9, 0.6]

        results = [_make_result(f"d{i}", f"doc {i}", 0.5, i) for i in range(3)]
        reranked = reranker.rerank("test", results, top_k=3)

        for i, r in enumerate(reranked):
            assert r.rank == i

    @patch("src.retrieval.reranker.CrossEncoderReranker._load_model")
    def test_builds_correct_pairs(self, mock_load: MagicMock) -> None:
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        reranker._model.predict.return_value = [0.5, 0.5]

        results = [
            _make_result("d1", "content A", 0.5, 0),
            _make_result("d2", "content B", 0.4, 1),
        ]
        reranker.rerank("my query", results, top_k=2)

        # Verify the pairs passed to predict
        call_args = reranker._model.predict.call_args[0][0]
        assert call_args == [["my query", "content A"], ["my query", "content B"]]

    def test_lazy_model_loading(self) -> None:
        reranker = CrossEncoderReranker()
        assert reranker._model is None

    @patch("src.retrieval.reranker.CrossEncoderReranker._load_model")
    def test_custom_model_name(self, mock_load: MagicMock) -> None:
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        assert reranker._model_name == "cross-encoder/ms-marco-TinyBERT-L-2-v2"


class TestLLMReranker:
    @patch("src.retrieval.reranker._call_llm")
    def test_rerank_sorts_by_llm_score(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "3,9,1"

        reranker = LLMReranker()
        results = [
            _make_result("d1", "doc one", 0.5, 0),
            _make_result("d2", "doc two", 0.4, 1),
            _make_result("d3", "doc three", 0.3, 2),
        ]

        reranked = reranker.rerank("test query", results, top_k=3)

        assert reranked[0].document.doc_id == "d2"  # Score 9
        assert reranked[1].document.doc_id == "d1"  # Score 3
        assert reranked[2].document.doc_id == "d3"  # Score 1

    @patch("src.retrieval.reranker._call_llm")
    def test_rerank_respects_top_k(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "8,5,3,9,2"

        reranker = LLMReranker()
        results = [_make_result(f"d{i}", f"doc {i}", 0.5, i) for i in range(5)]
        reranked = reranker.rerank("test", results, top_k=2)

        assert len(reranked) == 2

    @patch("src.retrieval.reranker._call_llm")
    def test_rerank_empty_input(self, mock_llm: MagicMock) -> None:
        reranker = LLMReranker()
        reranked = reranker.rerank("test", [], top_k=5)
        assert reranked == []
        mock_llm.assert_not_called()

    @patch("src.retrieval.reranker._call_llm")
    def test_scores_clamped_to_1_10(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "0,15"  # Out of range

        reranker = LLMReranker()
        results = [
            _make_result("d1", "doc one", 0.5, 0),
            _make_result("d2", "doc two", 0.4, 1),
        ]
        reranked = reranker.rerank("test", results, top_k=2)

        assert reranked[0].score == 10.0  # Clamped from 15
        assert reranked[1].score == 1.0  # Clamped from 0

    @patch("src.retrieval.reranker._call_llm")
    def test_handles_parse_failure(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "not,numbers,here"

        reranker = LLMReranker()
        results = [_make_result(f"d{i}", f"doc {i}", 0.5, i) for i in range(3)]
        reranked = reranker.rerank("test", results, top_k=3)

        # All should get default score of 5.0
        for r in reranked:
            assert r.score == 5.0

    @patch("src.retrieval.reranker._call_llm")
    def test_handles_too_few_scores(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "8"  # Only 1 score for 3 docs

        reranker = LLMReranker()
        results = [_make_result(f"d{i}", f"doc {i}", 0.5, i) for i in range(3)]
        reranked = reranker.rerank("test", results, top_k=3)

        assert len(reranked) == 3
        assert reranked[0].score == 8.0
        # Others padded with 5.0
        assert reranked[1].score == 5.0

    @patch("src.retrieval.reranker._call_llm")
    def test_batch_processing(self, mock_llm: MagicMock) -> None:
        # batch_size=2 means 3 docs → 2 LLM calls
        mock_llm.side_effect = ["8,5", "3"]

        reranker = LLMReranker(batch_size=2)
        results = [_make_result(f"d{i}", f"doc {i}", 0.5, i) for i in range(3)]
        reranked = reranker.rerank("test", results, top_k=3)

        assert mock_llm.call_count == 2
        assert len(reranked) == 3

    @patch("src.retrieval.reranker._call_llm")
    def test_ranks_are_sequential(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "7,3,9"

        reranker = LLMReranker()
        results = [_make_result(f"d{i}", f"doc {i}", 0.5, i) for i in range(3)]
        reranked = reranker.rerank("test", results, top_k=3)

        for i, r in enumerate(reranked):
            assert r.rank == i

    @patch("src.retrieval.reranker._call_llm")
    def test_document_content_truncated_in_prompt(self, mock_llm: MagicMock) -> None:
        mock_llm.return_value = "8"

        reranker = LLMReranker()
        long_content = "x" * 1000
        results = [_make_result("d1", long_content, 0.5, 0)]
        reranker.rerank("test", results, top_k=1)

        # Verify the prompt contains truncated content (500 chars)
        prompt = mock_llm.call_args[0][0]
        assert "x" * 500 in prompt
        assert "x" * 501 not in prompt
