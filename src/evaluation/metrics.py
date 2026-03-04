"""RAG evaluation metrics: faithfulness, answer relevancy, context precision, context recall.

Uses custom LLM-based scoring with structured prompts. Each metric returns
a score in [0, 1] plus a human-readable explanation.  Aggregate statistics
(mean, std, min, max) are computed over a batch of per-question results.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Protocol

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class LLMEvaluator(Protocol):
    """Minimal interface – anything with a ``generate`` method works."""

    def generate(self, prompt: str, system: str | None = None) -> str: ...


# ---------------------------------------------------------------------------
# Per-question metric result
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""

    name: str
    score: float  # 0.0 – 1.0
    explanation: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "score": self.score, "explanation": self.explanation}


@dataclass
class QuestionMetrics:
    """All metrics for one question."""

    question: str
    answer: str
    ground_truth: str
    contexts: list[str]
    faithfulness: MetricResult | None = None
    answer_relevancy: MetricResult | None = None
    context_precision: MetricResult | None = None
    context_recall: MetricResult | None = None

    def scores_dict(self) -> dict[str, float | None]:
        return {
            "faithfulness": self.faithfulness.score if self.faithfulness else None,
            "answer_relevancy": self.answer_relevancy.score if self.answer_relevancy else None,
            "context_precision": self.context_precision.score if self.context_precision else None,
            "context_recall": self.context_recall.score if self.context_recall else None,
        }

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "scores": self.scores_dict(),
            "faithfulness_detail": self.faithfulness.to_dict() if self.faithfulness else None,
            "answer_relevancy_detail": (
                self.answer_relevancy.to_dict() if self.answer_relevancy else None
            ),
            "context_precision_detail": (
                self.context_precision.to_dict() if self.context_precision else None
            ),
            "context_recall_detail": (
                self.context_recall.to_dict() if self.context_recall else None
            ),
        }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


@dataclass
class AggregateMetrics:
    """Aggregate (mean, std, min, max) over a batch of QuestionMetrics."""

    faithfulness: dict[str, float] = field(default_factory=dict)
    answer_relevancy: dict[str, float] = field(default_factory=dict)
    context_precision: dict[str, float] = field(default_factory=dict)
    context_recall: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


def compute_aggregate(results: list[QuestionMetrics]) -> AggregateMetrics:
    """Compute aggregate stats from a list of per-question metrics."""
    agg = AggregateMetrics()

    for metric_name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        scores = [
            getattr(r, metric_name).score for r in results if getattr(r, metric_name) is not None
        ]
        if scores:
            setattr(
                agg,
                metric_name,
                _summarise(scores),
            )

    return agg


def _summarise(scores: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.mean(scores), 4),
        "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "count": len(scores),
    }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of an answer to a question.
Faithfulness measures whether ALL claims in the answer can be inferred from the provided contexts.

Question: {question}

Contexts:
{contexts}

Answer: {answer}

Instructions:
1. List each claim in the answer.
2. For each claim, determine if it is supported by the contexts.
3. Calculate the fraction of supported claims.

Respond with ONLY a JSON object (no markdown fences):
{{"score": <float 0-1>, "explanation": "<brief explanation>"}}"""

_ANSWER_RELEVANCY_PROMPT = """You are evaluating the relevancy of an answer to a question.
Answer relevancy measures how well the answer addresses the original question.

Question: {question}

Answer: {answer}

Instructions:
1. Does the answer directly address the question?
2. Is the answer complete and on-topic?
3. Penalise off-topic or redundant information.

Respond with ONLY a JSON object (no markdown fences):
{{"score": <float 0-1>, "explanation": "<brief explanation>"}}"""

_CONTEXT_PRECISION_PROMPT = """You are evaluating context precision for a RAG system.
Context precision measures whether the retrieved contexts are relevant to the question.

Question: {question}

Contexts:
{contexts}

Ground truth answer: {ground_truth}

Instructions:
1. For each context, determine if it is relevant to answering the question correctly.
2. Calculate the fraction of contexts that are relevant.

Respond with ONLY a JSON object (no markdown fences):
{{"score": <float 0-1>, "explanation": "<brief explanation>"}}"""

_CONTEXT_RECALL_PROMPT = """You are evaluating context recall for a RAG system.
Context recall measures whether all information needed to answer the question is present in the contexts.

Question: {question}

Contexts:
{contexts}

Ground truth answer: {ground_truth}

Instructions:
1. Identify the key claims/facts in the ground truth answer.
2. For each claim, check if supporting information exists in the contexts.
3. Calculate the fraction of ground truth claims that are supported by the contexts.

Respond with ONLY a JSON object (no markdown fences):
{{"score": <float 0-1>, "explanation": "<brief explanation>"}}"""


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------


def _parse_llm_score(raw: str) -> tuple[float, str]:
    """Extract score and explanation from the LLM's JSON response."""
    # Try plain JSON first
    try:
        data = json.loads(raw)
        return _clamp(float(data["score"])), str(data.get("explanation", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Try stripping markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    try:
        data = json.loads(cleaned)
        return _clamp(float(data["score"])), str(data.get("explanation", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Last resort: regex
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', raw)
    if score_match:
        return _clamp(float(score_match.group(1))), "parsed via regex fallback"

    logger.warning("metric_parse_failed", raw_response=raw[:200])
    return 0.0, "failed to parse LLM response"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _format_contexts(contexts: list[str]) -> str:
    parts = []
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"[Context {i}]: {ctx}")
    return "\n\n".join(parts)


class RAGMetrics:
    """Evaluate RAG pipeline quality using LLM-as-judge scoring.

    Each metric is scored 0-1 with a natural language explanation.
    Falls back gracefully when the LLM returns un-parseable output.
    """

    def __init__(self, llm: LLMEvaluator) -> None:
        self._llm = llm

    def evaluate_faithfulness(
        self, question: str, answer: str, contexts: list[str]
    ) -> MetricResult:
        prompt = _FAITHFULNESS_PROMPT.format(
            question=question,
            answer=answer,
            contexts=_format_contexts(contexts),
        )
        raw = self._llm.generate(prompt)
        score, explanation = _parse_llm_score(raw)
        return MetricResult(name="faithfulness", score=score, explanation=explanation)

    def evaluate_answer_relevancy(self, question: str, answer: str) -> MetricResult:
        prompt = _ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
        raw = self._llm.generate(prompt)
        score, explanation = _parse_llm_score(raw)
        return MetricResult(name="answer_relevancy", score=score, explanation=explanation)

    def evaluate_context_precision(
        self, question: str, contexts: list[str], ground_truth: str
    ) -> MetricResult:
        prompt = _CONTEXT_PRECISION_PROMPT.format(
            question=question,
            contexts=_format_contexts(contexts),
            ground_truth=ground_truth,
        )
        raw = self._llm.generate(prompt)
        score, explanation = _parse_llm_score(raw)
        return MetricResult(name="context_precision", score=score, explanation=explanation)

    def evaluate_context_recall(
        self, question: str, contexts: list[str], ground_truth: str
    ) -> MetricResult:
        prompt = _CONTEXT_RECALL_PROMPT.format(
            question=question,
            contexts=_format_contexts(contexts),
            ground_truth=ground_truth,
        )
        raw = self._llm.generate(prompt)
        score, explanation = _parse_llm_score(raw)
        return MetricResult(name="context_recall", score=score, explanation=explanation)

    def evaluate_all(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> QuestionMetrics:
        """Run all four metrics for a single question."""
        qm = QuestionMetrics(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            contexts=contexts,
        )

        try:
            qm.faithfulness = self.evaluate_faithfulness(question, answer, contexts)
        except Exception as e:
            logger.warning("faithfulness_eval_failed", error=str(e))

        try:
            qm.answer_relevancy = self.evaluate_answer_relevancy(question, answer)
        except Exception as e:
            logger.warning("answer_relevancy_eval_failed", error=str(e))

        try:
            qm.context_precision = self.evaluate_context_precision(question, contexts, ground_truth)
        except Exception as e:
            logger.warning("context_precision_eval_failed", error=str(e))

        try:
            qm.context_recall = self.evaluate_context_recall(question, contexts, ground_truth)
        except Exception as e:
            logger.warning("context_recall_eval_failed", error=str(e))

        return qm
