"""RAG evaluation framework: metrics, datasets, and runner."""

from src.evaluation.dataset import EvalDataset, QAPair
from src.evaluation.metrics import (
    AggregateMetrics,
    MetricResult,
    QuestionMetrics,
    RAGMetrics,
    compute_aggregate,
)
from src.evaluation.runner import ComparisonResult, EvalReport, EvalRunner, compare_reports

__all__ = [
    "AggregateMetrics",
    "ComparisonResult",
    "EvalDataset",
    "EvalReport",
    "EvalRunner",
    "MetricResult",
    "QAPair",
    "QuestionMetrics",
    "RAGMetrics",
    "compare_reports",
    "compute_aggregate",
]
