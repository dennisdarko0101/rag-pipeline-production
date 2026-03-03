# Evaluation Framework

## Overview

The RAG evaluation framework uses **LLM-as-judge** scoring to measure four core metrics across faithfulness, relevancy, and context quality. A golden dataset of 18 hand-crafted Q&A pairs (based on the sample documents) provides repeatable benchmarks, and the evaluation runner produces structured reports with per-question and aggregate statistics.

## Architecture

```
┌──────────────────┐     ┌───────────────┐     ┌──────────────────┐
│   EvalDataset    │────▶│   EvalRunner  │────▶│   EvalReport     │
│  (Q&A pairs)     │     │               │     │  - per-question  │
│  - 18 golden     │     │  For each Q:  │     │  - aggregate     │
│  - categories    │     │  1. RAG query │     │  - latency       │
│  - load/save     │     │  2. Score     │     │  - JSON/Markdown  │
└──────────────────┘     │  3. Aggregate │     └──────────────────┘
                         └───────┬───────┘
                                 │
                         ┌───────▼───────┐
                         │  RAGMetrics   │
                         │  (LLM judge)  │
                         │               │
                         │  4 metrics:   │
                         │  faithfulness │
                         │  relevancy   │
                         │  precision   │
                         │  recall      │
                         └───────────────┘
```

## Metrics

All metrics are scored **0.0 – 1.0** with a natural-language explanation.

| Metric | What It Measures | How It's Computed |
|--------|-----------------|-------------------|
| **Faithfulness** | Are all claims in the answer supported by the retrieved contexts? | LLM lists claims in the answer, checks each against contexts, returns the fraction supported |
| **Answer Relevancy** | Does the answer address the original question? | LLM evaluates completeness and relevance, penalizes off-topic content |
| **Context Precision** | Are the retrieved contexts relevant to the question? | LLM checks each context for relevance to the question and ground truth |
| **Context Recall** | Is all information needed to answer present in the contexts? | LLM identifies key facts in the ground truth, checks if contexts contain supporting information |

### Aggregate Statistics

For a batch of questions, we compute:

| Statistic | Description |
|-----------|-------------|
| **Mean** | Average score across all questions |
| **Std** | Standard deviation (consistency indicator) |
| **Min** | Worst-case score |
| **Max** | Best-case score |
| **Count** | Number of questions scored |

## Golden Dataset

Located at `tests/eval/eval_dataset.json` — **18 Q&A pairs** across 4 categories:

| Category | Count | Description |
|----------|-------|-------------|
| **Straightforward** | 10 | Single-source questions with clear answers |
| **Multi-chunk** | 3 | Questions requiring information from multiple documents |
| **Unanswerable** | 3 | Questions not answerable from the corpus |
| **Adversarial** | 2 | Questions with false premises to test robustness |

### Source Documents

All Q&A pairs are based on the sample documents in `data/sample_docs/`:

- `rag_systems.md` — RAG architecture, chunking, hybrid search, evaluation
- `ai_agents.md` — Agent patterns (ReAct, Plan-and-Execute), memory types
- `transformer_architecture.md` — Self-attention, positional encoding, scaling laws
- `mlops_best_practices.md` — MLOps lifecycle, experiment tracking, deployment

### Dataset Format

```json
{
  "version": "1.0",
  "total_pairs": 18,
  "categories": {
    "straightforward": 10,
    "multi_chunk": 3,
    "unanswerable": 3,
    "adversarial": 2
  },
  "pairs": [
    {
      "question": "What are the five core components of a RAG system?",
      "ground_truth": "The five core components are...",
      "contexts": [],
      "category": "straightforward",
      "metadata": {"source_doc": "rag_systems.md", "difficulty": "easy"}
    }
  ]
}
```

## Running Evaluations

### Quick Start

```bash
# Run full evaluation with all defaults
make eval

# Or directly:
bash scripts/run_eval.sh
```

### Options

```bash
# Use a specific LLM provider
bash scripts/run_eval.sh --provider claude

# Evaluate only straightforward questions
bash scripts/run_eval.sh --category straightforward

# Custom dataset
bash scripts/run_eval.sh --dataset path/to/custom_dataset.json

# Adjust retrieval parameters
bash scripts/run_eval.sh --k 20 --rerank-top-k 10
```

### Output

Results are saved to `eval_results/`:

- `eval-YYYYMMDD-HHMMSS.json` — Full report with per-question scores and explanations
- `eval-YYYYMMDD-HHMMSS.md` — Human-readable Markdown summary
- `latest_scores.json` — Aggregate scores for CI threshold checks

### Console Output

```
============================================================
 EVALUATION RESULTS
============================================================
  faithfulness               0.8500 (+/- 0.1200)
  answer_relevancy           0.8800 (+/- 0.0900)
  context_precision          0.7600 (+/- 0.1500)
  context_recall             0.8100 (+/- 0.1100)

  Average latency:          2345.6 ms
  Total latency:            42221.0 ms
  Questions evaluated:      18
============================================================
```

## Comparing Runs

Use the `compare_reports()` function to track improvements and regressions between evaluation runs:

```python
from src.evaluation.runner import compare_reports, EvalReport
import json
from pathlib import Path

baseline = EvalReport(**json.loads(Path("eval_results/baseline.json").read_text()))
candidate = EvalReport(**json.loads(Path("eval_results/candidate.json").read_text()))

comparison = compare_reports(baseline, candidate, threshold=0.01)

print(comparison.summary)
# +0.0500 faithfulness
# -0.0200 context_precision
# unchanged: answer_relevancy, context_recall
```

## CI Integration

The `.github/workflows/eval.yml` workflow runs evaluations:

- **Scheduled:** Weekly on Sundays at 2 AM UTC
- **Manual:** Trigger via GitHub Actions with provider and category options
- **Quality gates:** Fails if faithfulness < 0.7 or answer relevancy < 0.7
- **Artifacts:** Results are uploaded and retained for 90 days

## Quality Thresholds

| Metric | CI Threshold | Target |
|--------|-------------|--------|
| Faithfulness | > 0.70 (fail CI) | > 0.85 |
| Answer Relevancy | > 0.70 (fail CI) | > 0.80 |
| Context Precision | — (monitored) | > 0.70 |
| Context Recall | — (monitored) | > 0.80 |

## Python API

### Core Classes

```python
from src.evaluation.metrics import RAGMetrics, MetricResult, QuestionMetrics
from src.evaluation.dataset import EvalDataset, QAPair
from src.evaluation.runner import EvalRunner, EvalReport, compare_reports

# Load dataset
dataset = EvalDataset.load("tests/eval/eval_dataset.json")

# Filter by category
hard_questions = dataset.filter_by_category("multi_chunk")

# Set up metrics (any LLM with .generate() method works)
from src.generation.llm import LLMFactory
llm = LLMFactory.create("claude")
metrics = RAGMetrics(llm=llm)

# Evaluate a single question
result = metrics.evaluate_all(
    question="What is RAG?",
    answer="RAG is...",
    contexts=["context 1", "context 2"],
    ground_truth="RAG is a technique that..."
)

print(result.scores_dict())
# {'faithfulness': 0.85, 'answer_relevancy': 0.9, ...}

# Run full evaluation
runner = EvalRunner(metrics=metrics, k=10, rerank_top_k=5)
report = runner.run(dataset, chain)

# Export
report.save_json("eval_results/report.json")
report.save_markdown("eval_results/report.md")
```

## Design Decisions

### Why LLM-as-Judge Instead of RAGAS Library Directly?

While RAGAS (ragas>=0.2.0) is included as a dependency, the evaluation framework uses custom LLM-based scoring for several reasons:

1. **Stability** — RAGAS API changes frequently across versions; custom prompts provide a stable interface
2. **Transparency** — Each metric has a clear prompt template that can be inspected and modified
3. **Flexibility** — Works with any LLM that has a `generate()` method (Claude, GPT, local models)
4. **Graceful degradation** — If one metric fails (LLM error), others still complete
5. **Explainability** — Each score includes a natural-language explanation

### Why Per-Question Error Isolation?

Each metric evaluation is wrapped in a try/except. If the LLM fails to score faithfulness for one question, the runner continues to score answer relevancy, context precision, and context recall. This maximizes the amount of useful data collected even when individual evaluations fail.

### Why JSON + Markdown Export?

- **JSON** for programmatic consumption (CI threshold checks, trend analysis, dashboards)
- **Markdown** for human review (pull requests, team communication, stakeholder reports)
