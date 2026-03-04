"""Unit tests for the evaluation runner, report, and comparison."""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

from src.evaluation.dataset import EvalDataset, QAPair
from src.evaluation.metrics import (
    AggregateMetrics,
    MetricResult,
    QuestionMetrics,
    RAGMetrics,
)
from src.evaluation.runner import ComparisonResult, EvalReport, EvalRunner, compare_reports

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeSource:
    source_name: str = "doc.md"
    chunk_text: str = "chunk content"
    chunk_index: int = 0
    relevance_score: float = 0.9


@dataclass
class _FakeRAGResponse:
    answer: str = "Test answer."
    sources: list = field(default_factory=lambda: [_FakeSource()])
    citations: list = field(default_factory=list)
    metadata: dict = field(default_factory=lambda: {"latency_ms": 10.0})


def _make_mock_chain(response: _FakeRAGResponse | None = None) -> MagicMock:
    chain = MagicMock()
    chain.query.return_value = response or _FakeRAGResponse()
    return chain


def _make_mock_metrics(score: float = 0.8) -> RAGMetrics:
    """Create a RAGMetrics backed by a mock LLM that returns a fixed score."""
    llm = MagicMock()
    llm.generate.return_value = json.dumps({"score": score, "explanation": "mock"})
    return RAGMetrics(llm=llm)


def _make_dataset(n: int = 3) -> EvalDataset:
    pairs = [
        QAPair(
            question=f"Question {i}?",
            ground_truth=f"Answer {i}.",
            category="straightforward",
        )
        for i in range(n)
    ]
    return EvalDataset(pairs=pairs)


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------


class TestEvalRunner:
    def test_run_returns_report(self) -> None:
        chain = _make_mock_chain()
        metrics = _make_mock_metrics(0.85)
        runner = EvalRunner(metrics=metrics, k=5, rerank_top_k=3)
        dataset = _make_dataset(2)

        report = runner.run(dataset, chain, run_id="test-001")

        assert report.run_id == "test-001"
        assert report.total_questions == 2
        assert len(report.per_question) == 2
        assert report.avg_latency_ms >= 0

    def test_run_calls_chain_for_each_question(self) -> None:
        chain = _make_mock_chain()
        metrics = _make_mock_metrics()
        runner = EvalRunner(metrics=metrics, k=10, rerank_top_k=5)
        dataset = _make_dataset(4)

        runner.run(dataset, chain)

        assert chain.query.call_count == 4

    def test_run_passes_k_and_rerank_top_k(self) -> None:
        chain = _make_mock_chain()
        metrics = _make_mock_metrics()
        runner = EvalRunner(metrics=metrics, k=20, rerank_top_k=8)
        dataset = _make_dataset(1)

        runner.run(dataset, chain)

        chain.query.assert_called_once_with(question="Question 0?", k=20, rerank_top_k=8)

    def test_run_handles_chain_error(self) -> None:
        chain = MagicMock()
        chain.query.side_effect = RuntimeError("Chain exploded")
        metrics = _make_mock_metrics()
        runner = EvalRunner(metrics=metrics)
        dataset = _make_dataset(1)

        report = runner.run(dataset, chain)

        assert report.total_questions == 1
        # Should still produce metrics (on the error answer)
        assert len(report.per_question) == 1

    def test_run_uses_dataset_contexts_when_chain_returns_none(self) -> None:
        chain = _make_mock_chain(_FakeRAGResponse(sources=[]))
        llm = MagicMock()
        llm.generate.return_value = json.dumps({"score": 0.7, "explanation": "ok"})
        metrics = RAGMetrics(llm=llm)
        runner = EvalRunner(metrics=metrics)

        pair = QAPair(
            question="Q?",
            ground_truth="GT.",
            contexts=["predefined context"],
        )
        dataset = EvalDataset(pairs=[pair])

        runner.run(dataset, chain)

        # Verify the predefined context was used in the metrics prompt
        calls = llm.generate.call_args_list
        # Faithfulness prompt should contain the predefined context
        assert any("predefined context" in str(call) for call in calls)

    def test_run_auto_generates_run_id(self) -> None:
        chain = _make_mock_chain()
        metrics = _make_mock_metrics()
        runner = EvalRunner(metrics=metrics)
        dataset = _make_dataset(1)

        report = runner.run(dataset, chain)

        assert report.run_id.startswith("eval-")

    def test_report_config_captured(self) -> None:
        chain = _make_mock_chain()
        metrics = _make_mock_metrics()
        runner = EvalRunner(metrics=metrics, k=15, rerank_top_k=7)
        dataset = _make_dataset(1)

        report = runner.run(dataset, chain)

        assert report.config["k"] == 15
        assert report.config["rerank_top_k"] == 7


# ---------------------------------------------------------------------------
# EvalReport serialisation
# ---------------------------------------------------------------------------


class TestEvalReport:
    @staticmethod
    def _make_report() -> EvalReport:
        qm = QuestionMetrics(
            question="Q?",
            answer="A.",
            ground_truth="GT.",
            contexts=["c"],
            faithfulness=MetricResult("faithfulness", 0.8, "Good"),
            answer_relevancy=MetricResult("answer_relevancy", 0.9, "Relevant"),
        )
        agg = AggregateMetrics(
            faithfulness={"mean": 0.8, "std": 0.0, "min": 0.8, "max": 0.8, "count": 1},
            answer_relevancy={"mean": 0.9, "std": 0.0, "min": 0.9, "max": 0.9, "count": 1},
        )
        return EvalReport(
            run_id="test-001",
            timestamp="2025-01-01T00:00:00+00:00",
            total_questions=1,
            per_question=[qm],
            aggregate=agg,
            avg_latency_ms=42.5,
            total_latency_ms=42.5,
            config={"k": 10},
        )

    def test_to_dict(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        assert d["run_id"] == "test-001"
        assert d["total_questions"] == 1
        assert d["aggregate"]["faithfulness"]["mean"] == 0.8
        assert len(d["per_question"]) == 1

    def test_save_json(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save_json(path)

            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["run_id"] == "test-001"
            assert data["aggregate"]["faithfulness"]["mean"] == 0.8

    def test_save_markdown(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.save_markdown(path)

            md = path.read_text(encoding="utf-8")
            assert "# Evaluation Report" in md
            assert "test-001" in md
            assert "faithfulness" in md
            assert "0.8000" in md

    def test_save_creates_parent_dirs(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "report.json"
            report.save_json(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# compare_reports
# ---------------------------------------------------------------------------


class TestCompareReports:
    @staticmethod
    def _make_report_with_scores(run_id: str, faith: float, relevancy: float) -> EvalReport:
        agg = AggregateMetrics(
            faithfulness={"mean": faith, "std": 0.0, "min": faith, "max": faith, "count": 1},
            answer_relevancy={
                "mean": relevancy,
                "std": 0.0,
                "min": relevancy,
                "max": relevancy,
                "count": 1,
            },
        )
        return EvalReport(
            run_id=run_id,
            timestamp="2025-01-01",
            total_questions=1,
            per_question=[],
            aggregate=agg,
            avg_latency_ms=10.0,
            total_latency_ms=10.0,
        )

    def test_improvement_detected(self) -> None:
        baseline = self._make_report_with_scores("base", 0.7, 0.7)
        candidate = self._make_report_with_scores("cand", 0.85, 0.7)

        result = compare_reports(baseline, candidate)

        assert "faithfulness" in result.improvements
        assert result.improvements["faithfulness"] > 0

    def test_regression_detected(self) -> None:
        baseline = self._make_report_with_scores("base", 0.9, 0.9)
        candidate = self._make_report_with_scores("cand", 0.7, 0.9)

        result = compare_reports(baseline, candidate)

        assert "faithfulness" in result.regressions
        assert result.regressions["faithfulness"] < 0

    def test_unchanged_detected(self) -> None:
        baseline = self._make_report_with_scores("base", 0.8, 0.8)
        candidate = self._make_report_with_scores("cand", 0.805, 0.8)

        result = compare_reports(baseline, candidate, threshold=0.01)

        assert "faithfulness" in result.unchanged
        assert "answer_relevancy" in result.unchanged

    def test_comparison_result_summary(self) -> None:
        result = ComparisonResult(
            baseline_id="base",
            candidate_id="cand",
            improvements={"faithfulness": 0.1},
            regressions={"context_precision": -0.05},
            unchanged=["answer_relevancy"],
        )
        summary = result.summary
        assert "faithfulness" in summary
        assert "context_precision" in summary
        assert "answer_relevancy" in summary

    def test_comparison_to_dict(self) -> None:
        result = ComparisonResult(
            baseline_id="base",
            candidate_id="cand",
            improvements={"faithfulness": 0.1},
        )
        d = result.to_dict()
        assert d["baseline_id"] == "base"
        assert d["improvements"]["faithfulness"] == 0.1

    def test_comparison_with_missing_metrics(self) -> None:
        """If one report has a metric and the other doesn't, skip it."""
        baseline = EvalReport(
            run_id="base",
            timestamp="2025-01-01",
            total_questions=0,
            per_question=[],
            aggregate=AggregateMetrics(faithfulness={"mean": 0.8, "count": 1}),
            avg_latency_ms=0,
            total_latency_ms=0,
        )
        candidate = EvalReport(
            run_id="cand",
            timestamp="2025-01-01",
            total_questions=0,
            per_question=[],
            aggregate=AggregateMetrics(),  # No metrics at all
            avg_latency_ms=0,
            total_latency_ms=0,
        )
        result = compare_reports(baseline, candidate)
        # Should not crash, no improvements or regressions
        assert result.improvements == {}
        assert result.regressions == {}


# ---------------------------------------------------------------------------
# EvalDataset integration with runner
# ---------------------------------------------------------------------------


class TestDatasetRunnerIntegration:
    def test_filtered_dataset_works_with_runner(self) -> None:
        pairs = [
            QAPair(question="Easy Q?", ground_truth="Easy A.", category="straightforward"),
            QAPair(question="Hard Q?", ground_truth="Hard A.", category="multi_chunk"),
        ]
        dataset = EvalDataset(pairs=pairs)
        filtered = dataset.filter_by_category("straightforward")

        chain = _make_mock_chain()
        metrics = _make_mock_metrics()
        runner = EvalRunner(metrics=metrics)

        report = runner.run(filtered, chain)

        assert report.total_questions == 1
        assert chain.query.call_count == 1
