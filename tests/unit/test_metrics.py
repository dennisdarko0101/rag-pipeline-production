"""Unit tests for the RAG evaluation metrics module."""

import json
from unittest.mock import MagicMock

from src.evaluation.metrics import (
    AggregateMetrics,
    MetricResult,
    QuestionMetrics,
    RAGMetrics,
    _clamp,
    _parse_llm_score,
    compute_aggregate,
)

# ---------------------------------------------------------------------------
# _parse_llm_score
# ---------------------------------------------------------------------------


class TestParseLLMScore:
    """Tests for the LLM JSON response parser."""

    def test_valid_json(self) -> None:
        raw = json.dumps({"score": 0.85, "explanation": "Good answer."})
        score, explanation = _parse_llm_score(raw)
        assert score == 0.85
        assert explanation == "Good answer."

    def test_json_with_markdown_fences(self) -> None:
        raw = '```json\n{"score": 0.72, "explanation": "Decent."}\n```'
        score, explanation = _parse_llm_score(raw)
        assert score == 0.72
        assert explanation == "Decent."

    def test_regex_fallback(self) -> None:
        raw = 'Some preamble... "score": 0.60, "explanation": unparseable'
        score, explanation = _parse_llm_score(raw)
        assert score == 0.60
        assert "regex" in explanation.lower()

    def test_completely_unparseable(self) -> None:
        raw = "I cannot provide a score."
        score, explanation = _parse_llm_score(raw)
        assert score == 0.0
        assert "failed" in explanation.lower()

    def test_score_clamped_above_one(self) -> None:
        raw = json.dumps({"score": 1.5, "explanation": "Overconfident."})
        score, _ = _parse_llm_score(raw)
        assert score == 1.0

    def test_score_clamped_below_zero(self) -> None:
        raw = json.dumps({"score": -0.3, "explanation": "Negative."})
        score, _ = _parse_llm_score(raw)
        assert score == 0.0

    def test_score_no_explanation(self) -> None:
        raw = json.dumps({"score": 0.5})
        score, explanation = _parse_llm_score(raw)
        assert score == 0.5
        assert explanation == ""


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_at_boundaries(self) -> None:
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_above(self) -> None:
        assert _clamp(1.5) == 1.0

    def test_below(self) -> None:
        assert _clamp(-0.1) == 0.0


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


class TestMetricResult:
    def test_to_dict(self) -> None:
        mr = MetricResult(name="faithfulness", score=0.9, explanation="All good")
        d = mr.to_dict()
        assert d["name"] == "faithfulness"
        assert d["score"] == 0.9
        assert d["explanation"] == "All good"


# ---------------------------------------------------------------------------
# QuestionMetrics
# ---------------------------------------------------------------------------


class TestQuestionMetrics:
    def test_scores_dict_all_present(self) -> None:
        qm = QuestionMetrics(
            question="Q?",
            answer="A.",
            ground_truth="GT.",
            contexts=["ctx"],
            faithfulness=MetricResult("faithfulness", 0.8),
            answer_relevancy=MetricResult("answer_relevancy", 0.7),
            context_precision=MetricResult("context_precision", 0.9),
            context_recall=MetricResult("context_recall", 0.6),
        )
        d = qm.scores_dict()
        assert d["faithfulness"] == 0.8
        assert d["answer_relevancy"] == 0.7
        assert d["context_precision"] == 0.9
        assert d["context_recall"] == 0.6

    def test_scores_dict_missing_metrics(self) -> None:
        qm = QuestionMetrics(question="Q?", answer="A.", ground_truth="GT.", contexts=[])
        d = qm.scores_dict()
        assert d["faithfulness"] is None
        assert d["answer_relevancy"] is None

    def test_to_dict_includes_all_fields(self) -> None:
        qm = QuestionMetrics(
            question="Q?",
            answer="A.",
            ground_truth="GT.",
            contexts=["ctx"],
            faithfulness=MetricResult("faithfulness", 0.8, "ok"),
        )
        d = qm.to_dict()
        assert d["question"] == "Q?"
        assert d["faithfulness_detail"]["score"] == 0.8
        assert d["answer_relevancy_detail"] is None


# ---------------------------------------------------------------------------
# compute_aggregate
# ---------------------------------------------------------------------------


class TestComputeAggregate:
    def test_single_result(self) -> None:
        qm = QuestionMetrics(
            question="Q?",
            answer="A.",
            ground_truth="GT.",
            contexts=["c"],
            faithfulness=MetricResult("faithfulness", 0.8),
            answer_relevancy=MetricResult("answer_relevancy", 0.9),
        )
        agg = compute_aggregate([qm])
        assert agg.faithfulness["mean"] == 0.8
        assert agg.faithfulness["std"] == 0.0  # single item
        assert agg.answer_relevancy["mean"] == 0.9

    def test_multiple_results(self) -> None:
        results = [
            QuestionMetrics(
                question="Q1",
                answer="A1",
                ground_truth="GT1",
                contexts=["c"],
                faithfulness=MetricResult("faithfulness", 0.6),
                answer_relevancy=MetricResult("answer_relevancy", 0.8),
            ),
            QuestionMetrics(
                question="Q2",
                answer="A2",
                ground_truth="GT2",
                contexts=["c"],
                faithfulness=MetricResult("faithfulness", 1.0),
                answer_relevancy=MetricResult("answer_relevancy", 0.4),
            ),
        ]
        agg = compute_aggregate(results)
        assert agg.faithfulness["mean"] == 0.8
        assert agg.faithfulness["min"] == 0.6
        assert agg.faithfulness["max"] == 1.0
        assert agg.faithfulness["count"] == 2
        assert agg.answer_relevancy["mean"] == 0.6

    def test_empty_results(self) -> None:
        agg = compute_aggregate([])
        assert agg.faithfulness == {}
        assert agg.answer_relevancy == {}

    def test_partial_metrics(self) -> None:
        """Some questions have metrics, others don't."""
        results = [
            QuestionMetrics(
                question="Q1",
                answer="A1",
                ground_truth="GT1",
                contexts=["c"],
                faithfulness=MetricResult("faithfulness", 0.7),
            ),
            QuestionMetrics(
                question="Q2",
                answer="A2",
                ground_truth="GT2",
                contexts=["c"],
            ),
        ]
        agg = compute_aggregate(results)
        assert agg.faithfulness["count"] == 1
        assert agg.faithfulness["mean"] == 0.7
        assert agg.answer_relevancy == {}

    def test_aggregate_to_dict(self) -> None:
        agg = AggregateMetrics(
            faithfulness={"mean": 0.8, "std": 0.1, "min": 0.6, "max": 1.0, "count": 3}
        )
        d = agg.to_dict()
        assert d["faithfulness"]["mean"] == 0.8
        assert d["answer_relevancy"] == {}


# ---------------------------------------------------------------------------
# RAGMetrics (LLM-as-judge)
# ---------------------------------------------------------------------------


class TestRAGMetrics:
    """Tests for RAGMetrics using a mock LLM."""

    @staticmethod
    def _mock_llm(score: float = 0.85, explanation: str = "Looks good") -> MagicMock:
        llm = MagicMock()
        llm.generate.return_value = json.dumps({"score": score, "explanation": explanation})
        return llm

    def test_evaluate_faithfulness(self) -> None:
        llm = self._mock_llm(0.9, "All claims supported")
        metrics = RAGMetrics(llm=llm)
        result = metrics.evaluate_faithfulness("Q?", "A.", ["ctx1", "ctx2"])
        assert result.name == "faithfulness"
        assert result.score == 0.9
        assert "supported" in result.explanation
        llm.generate.assert_called_once()

    def test_evaluate_answer_relevancy(self) -> None:
        llm = self._mock_llm(0.75, "Mostly relevant")
        metrics = RAGMetrics(llm=llm)
        result = metrics.evaluate_answer_relevancy("Q?", "A.")
        assert result.name == "answer_relevancy"
        assert result.score == 0.75

    def test_evaluate_context_precision(self) -> None:
        llm = self._mock_llm(0.8, "Most contexts relevant")
        metrics = RAGMetrics(llm=llm)
        result = metrics.evaluate_context_precision("Q?", ["c1", "c2"], "GT")
        assert result.name == "context_precision"
        assert result.score == 0.8

    def test_evaluate_context_recall(self) -> None:
        llm = self._mock_llm(0.65, "Some claims missing")
        metrics = RAGMetrics(llm=llm)
        result = metrics.evaluate_context_recall("Q?", ["c1"], "GT with many claims")
        assert result.name == "context_recall"
        assert result.score == 0.65

    def test_evaluate_all(self) -> None:
        llm = self._mock_llm(0.8, "Good")
        metrics = RAGMetrics(llm=llm)
        qm = metrics.evaluate_all("Q?", "A.", ["c1"], "GT")
        assert qm.faithfulness is not None
        assert qm.answer_relevancy is not None
        assert qm.context_precision is not None
        assert qm.context_recall is not None
        assert llm.generate.call_count == 4

    def test_evaluate_all_tolerates_individual_failure(self) -> None:
        """If one metric fails, others should still succeed."""
        call_count = 0

        def flaky_generate(prompt: str, system: str | None = None) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM failed")
            return json.dumps({"score": 0.7, "explanation": "ok"})

        llm = MagicMock()
        llm.generate.side_effect = flaky_generate

        metrics = RAGMetrics(llm=llm)
        qm = metrics.evaluate_all("Q?", "A.", ["c1"], "GT")

        # First metric (faithfulness) should be None due to error
        assert qm.faithfulness is None
        # Other metrics should succeed
        assert qm.answer_relevancy is not None
        assert qm.answer_relevancy.score == 0.7

    def test_prompt_contains_question(self) -> None:
        llm = self._mock_llm()
        metrics = RAGMetrics(llm=llm)
        metrics.evaluate_faithfulness("What is RAG?", "RAG is...", ["context"])
        call_args = llm.generate.call_args[0][0]
        assert "What is RAG?" in call_args

    def test_prompt_contains_contexts(self) -> None:
        llm = self._mock_llm()
        metrics = RAGMetrics(llm=llm)
        metrics.evaluate_faithfulness("Q?", "A.", ["alpha context", "beta context"])
        call_args = llm.generate.call_args[0][0]
        assert "alpha context" in call_args
        assert "beta context" in call_args
