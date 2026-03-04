"""Evaluation runner: orchestrate RAG evaluation and produce reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

from src.evaluation.metrics import (
    AggregateMetrics,
    QuestionMetrics,
    RAGMetrics,
    compute_aggregate,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from src.evaluation.dataset import EvalDataset, QAPair


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class RAGChainProtocol(Protocol):
    """Minimal RAGChain interface required by the runner."""

    def query(self, question: str, k: int = 10, rerank_top_k: int = 5) -> _RAGResponseLike: ...


class _RAGResponseLike(Protocol):
    answer: str
    sources: list
    metadata: dict


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Complete evaluation report with per-question and aggregate metrics."""

    run_id: str
    timestamp: str
    total_questions: int
    per_question: list[QuestionMetrics]
    aggregate: AggregateMetrics
    avg_latency_ms: float
    total_latency_ms: float
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_questions": self.total_questions,
            "aggregate": self.aggregate.to_dict(),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "config": self.config,
            "per_question": [q.to_dict() for q in self.per_question],
        }

    def save_json(self, path: str | Path) -> None:
        """Save report as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("report_saved_json", path=str(path))

    def save_markdown(self, path: str | Path) -> None:
        """Save report as a Markdown summary."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render_markdown(), encoding="utf-8")
        logger.info("report_saved_markdown", path=str(path))

    def _render_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Evaluation Report — {self.run_id}")
        lines.append("")
        lines.append(f"**Timestamp:** {self.timestamp}")
        lines.append(f"**Questions:** {self.total_questions}")
        lines.append(f"**Avg latency:** {self.avg_latency_ms:.1f} ms")
        lines.append(f"**Total latency:** {self.total_latency_ms:.1f} ms")
        lines.append("")

        # Aggregate table
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("|--------|------|-----|-----|-----|")
        for metric_name in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ):
            stats = getattr(self.aggregate, metric_name, {})
            if stats:
                lines.append(
                    f"| {metric_name} | {stats.get('mean', '-'):.4f} "
                    f"| {stats.get('std', '-'):.4f} "
                    f"| {stats.get('min', '-'):.4f} "
                    f"| {stats.get('max', '-'):.4f} |"
                )
        lines.append("")

        # Per-question summary
        lines.append("## Per-Question Results")
        lines.append("")
        lines.append("| # | Question | Faith. | Relev. | Prec. | Recall | Latency |")
        lines.append("|---|----------|--------|--------|-------|--------|---------|")
        for i, qm in enumerate(self.per_question, 1):
            f = f"{qm.faithfulness.score:.2f}" if qm.faithfulness else "-"
            r = f"{qm.answer_relevancy.score:.2f}" if qm.answer_relevancy else "-"
            p = f"{qm.context_precision.score:.2f}" if qm.context_precision else "-"
            c = f"{qm.context_recall.score:.2f}" if qm.context_recall else "-"
            q = qm.question[:50] + ("..." if len(qm.question) > 50 else "")
            latency = qm.scores_dict().get("latency_ms", "-")
            lines.append(f"| {i} | {q} | {f} | {r} | {p} | {c} | {latency} |")
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of comparing two evaluation reports."""

    baseline_id: str
    candidate_id: str
    improvements: dict[str, float] = field(default_factory=dict)
    regressions: dict[str, float] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def summary(self) -> str:
        parts = []
        if self.improvements:
            for m, delta in self.improvements.items():
                parts.append(f"  +{delta:+.4f} {m}")
        if self.regressions:
            for m, delta in self.regressions.items():
                parts.append(f"  {delta:+.4f} {m}")
        if self.unchanged:
            parts.append(f"  unchanged: {', '.join(self.unchanged)}")
        return "\n".join(parts) if parts else "  no metrics to compare"


def compare_reports(
    baseline: EvalReport,
    candidate: EvalReport,
    threshold: float = 0.01,
) -> ComparisonResult:
    """Compare two evaluation reports and categorize metric changes.

    Args:
        baseline: The reference report.
        candidate: The new report to compare.
        threshold: Minimum absolute change to count as improvement/regression.

    Returns:
        ComparisonResult with deltas.
    """
    result = ComparisonResult(
        baseline_id=baseline.run_id,
        candidate_id=candidate.run_id,
    )

    for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        b_stats = getattr(baseline.aggregate, metric_name, {})
        c_stats = getattr(candidate.aggregate, metric_name, {})

        b_mean = b_stats.get("mean") if b_stats else None
        c_mean = c_stats.get("mean") if c_stats else None

        if b_mean is None or c_mean is None:
            continue

        delta = c_mean - b_mean
        if delta > threshold:
            result.improvements[metric_name] = round(delta, 4)
        elif delta < -threshold:
            result.regressions[metric_name] = round(delta, 4)
        else:
            result.unchanged.append(metric_name)

    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvalRunner:
    """Orchestrates RAG evaluation over a dataset.

    Usage::

        runner = EvalRunner(metrics=rag_metrics, k=10)
        report = runner.run(dataset, chain)
        report.save_json("eval_results/report.json")
    """

    def __init__(
        self,
        metrics: RAGMetrics,
        k: int = 10,
        rerank_top_k: int = 5,
    ) -> None:
        self._metrics = metrics
        self._k = k
        self._rerank_top_k = rerank_top_k

    def run(
        self,
        dataset: EvalDataset,
        chain: RAGChainProtocol,  # type: ignore[valid-type]
        run_id: str | None = None,
    ) -> EvalReport:
        """Run evaluation over all Q&A pairs.

        Args:
            dataset: The evaluation dataset.
            chain: A RAGChain (or compatible) to evaluate.
            run_id: Optional identifier; auto-generated if omitted.

        Returns:
            Complete EvalReport.
        """
        run_id = run_id or datetime.now(timezone.utc).strftime("eval-%Y%m%d-%H%M%S")
        logger.info("eval_run_started", run_id=run_id, num_questions=len(dataset))

        total_start = perf_counter()
        results: list[QuestionMetrics] = []
        latencies: list[float] = []

        for i, pair in enumerate(dataset.pairs):
            logger.info("eval_question", index=i + 1, total=len(dataset), category=pair.category)
            qm, latency = self._evaluate_single(pair, chain)
            results.append(qm)
            latencies.append(latency)

        total_elapsed = (perf_counter() - total_start) * 1000
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        aggregate = compute_aggregate(results)

        report = EvalReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_questions=len(dataset),
            per_question=results,
            aggregate=aggregate,
            avg_latency_ms=avg_latency,
            total_latency_ms=total_elapsed,
            config={
                "k": self._k,
                "rerank_top_k": self._rerank_top_k,
            },
        )

        logger.info(
            "eval_run_completed",
            run_id=run_id,
            total_questions=len(dataset),
            avg_latency_ms=round(avg_latency, 1),
            aggregate=aggregate.to_dict(),
        )

        return report

    def _evaluate_single(
        self,
        pair: QAPair,
        chain: RAGChainProtocol,  # type: ignore[valid-type]
    ) -> tuple[QuestionMetrics, float]:
        """Evaluate a single Q&A pair: run RAG chain then score."""
        start = perf_counter()

        # Run the RAG chain
        try:
            response = chain.query(
                question=pair.question,
                k=self._k,
                rerank_top_k=self._rerank_top_k,
            )
            answer = response.answer
            contexts = [s.chunk_text for s in response.sources] if response.sources else []
        except Exception as e:
            logger.warning("eval_chain_error", question=pair.question[:60], error=str(e))
            answer = f"[ERROR] {e}"
            contexts = []

        latency_ms = (perf_counter() - start) * 1000

        # Use pre-defined contexts from the dataset if the chain returned none
        if not contexts and pair.contexts:
            contexts = pair.contexts

        # Score with metrics
        qm = self._metrics.evaluate_all(
            question=pair.question,
            answer=answer,
            contexts=contexts,
            ground_truth=pair.ground_truth,
        )

        return qm, latency_ms
