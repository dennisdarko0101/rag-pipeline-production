# Evaluation Methodology

## Overview

RAG systems need evaluation at multiple levels: retrieval quality, generation quality, and end-to-end performance. This document outlines the evaluation framework planned for Phase 3, using RAGAS as the primary evaluation library.

## Why Evaluate?

Without measurement, you can't improve. RAG systems have multiple failure modes:
- **Retrieval failure** -- relevant documents aren't found
- **Context overload** -- too many irrelevant documents dilute the signal
- **Hallucination** -- the LLM invents information not in the context
- **Citation errors** -- sources are misattributed or fabricated
- **Incomplete answers** -- the answer misses key information from retrieved docs

Each failure mode requires a different metric to detect.

## RAGAS Metrics

[RAGAS](https://docs.ragas.io/) (Retrieval Augmented Generation Assessment) provides a framework for evaluating RAG pipelines. We will track these core metrics:

### Retrieval Metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Context Precision** | Are the retrieved documents relevant to the question? | High precision means less noise in the context, leading to better answers. |
| **Context Recall** | Does the retrieved context contain all the information needed to answer? | High recall ensures the LLM has the full picture. |

### Generation Metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? | The core metric for grounding. Low faithfulness = hallucination. |
| **Answer Relevancy** | Does the answer actually address the question asked? | Prevents tangential or off-topic responses. |

### End-to-End Metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Answer Correctness** | Is the answer factually correct compared to ground truth? | The ultimate quality measure. |
| **Latency (p50, p95, p99)** | How long does the full pipeline take? | User experience and SLA compliance. |
| **Token Usage** | How many tokens are consumed per query? | Cost optimization. |

## Evaluation Dataset Design

### Structure

Each evaluation example consists of:

```python
{
    "question": "How does the attention mechanism work in transformers?",
    "ground_truth": "The attention mechanism computes...",
    "contexts": ["Retrieved chunk 1...", "Retrieved chunk 2..."],
    "answer": "Generated answer from the RAG pipeline..."
}
```

### Data Sources

1. **Sample documents** -- The 4 technical articles in `data/sample_docs/` with hand-written Q&A pairs
2. **Synthetic generation** -- LLM-generated questions from document chunks
3. **Edge cases** -- Questions that require multi-hop reasoning, questions with no answer in the corpus, ambiguous queries

### Planned Dataset Size

| Split | Size | Purpose |
|-------|------|---------|
| Core | 20-30 | Hand-crafted, high-quality Q&A pairs for reliable benchmarking |
| Extended | 100+ | Synthetically generated for stress testing |
| Adversarial | 10-15 | Edge cases: unanswerable, ambiguous, multi-hop |

## Pipeline-Level Evaluation

Beyond RAGAS metrics, we will measure each pipeline stage independently:

### Retrieval Evaluation

```
For each (question, ground_truth_chunks) pair:
  1. Run retrieval (semantic, BM25, hybrid)
  2. Measure Recall@k: what fraction of ground truth chunks are in top-k?
  3. Measure MRR: where does the first relevant chunk appear?
  4. Compare retriever configurations (weights, k values)
```

### Reranking Evaluation

```
For each (question, retrieved_results) pair:
  1. Run reranking (cross-encoder, LLM)
  2. Measure NDCG@k: are relevant docs ranked higher after reranking?
  3. Measure reranking latency vs. accuracy tradeoff
```

### Generation Evaluation

```
For each (question, context, ground_truth) triple:
  1. Generate answer via RAGChain
  2. Measure faithfulness (RAGAS)
  3. Measure answer relevancy (RAGAS)
  4. Validate citations against retrieved sources
  5. Measure citation accuracy: % of citations that are correct
```

## Running Evaluations

```bash
# Run the full evaluation suite
make eval

# Run specific evaluation tests
pytest tests/eval/ -v --tb=long
```

## Metrics We Track in Production

The RAGResponse metadata already captures per-query metrics:

| Metric | Source | Description |
|--------|--------|-------------|
| `latency_ms` | `RAGResponse.metadata` | Total pipeline latency |
| `retrieve_ms` | `RAGResponse.metadata` | Retrieval stage latency |
| `rerank_ms` | `RAGResponse.metadata` | Reranking stage latency |
| `generate_ms` | `RAGResponse.metadata` | LLM generation latency |
| `num_retrieved` | `RAGResponse.metadata` | Documents retrieved |
| `num_reranked` | `RAGResponse.metadata` | Documents after reranking |
| `num_citations` | `RAGResponse.metadata` | Valid citations in response |
| `tokens_used` | `RAGResponse.metadata` | Input/output token counts |
| `fallback_rate` | `FallbackLLM.fallback_stats` | How often the fallback LLM is used |

These metrics will be exported to Prometheus via the metrics defined in `src/utils/monitoring.py` for dashboard monitoring.

## Targets

Initial quality targets (to be refined after baseline measurement):

| Metric | Target | Rationale |
|--------|--------|-----------|
| Faithfulness | > 0.85 | Answers should be well-grounded |
| Answer Relevancy | > 0.80 | Answers should address the question |
| Context Precision | > 0.70 | Majority of retrieved docs should be relevant |
| Context Recall | > 0.80 | Most needed information should be retrieved |
| P95 Latency | < 5s | Acceptable for a conversational interface |
| Citation Accuracy | > 0.90 | Nearly all citations should be valid |

## Next Steps (Phase 3)

1. Create evaluation dataset with hand-crafted Q&A pairs for sample documents
2. Implement RAGAS evaluation runner in `tests/eval/`
3. Add retrieval-specific metrics (Recall@k, MRR)
4. Build evaluation reporting (JSON + HTML summary)
5. Integrate metrics into CI (fail on quality regression)
