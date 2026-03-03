#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# RAG Pipeline Evaluation Runner
# ---------------------------------------------------------------------------
# Usage:
#   bash scripts/run_eval.sh                       # full evaluation
#   bash scripts/run_eval.sh --category straightforward  # filter by category
#   bash scripts/run_eval.sh --dataset path/to.json      # custom dataset
#   bash scripts/run_eval.sh --provider openai           # specify LLM provider
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="${PROJECT_ROOT}/tests/eval/eval_dataset.json"
OUTPUT_DIR="${PROJECT_ROOT}/eval_results"
PROVIDER="fallback"
CATEGORY=""
K=10
RERANK_TOP_K=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --provider)   PROVIDER="$2"; shift 2 ;;
        --category)   CATEGORY="$2"; shift 2 ;;
        --k)          K="$2"; shift 2 ;;
        --rerank-top-k) RERANK_TOP_K="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset PATH        Path to evaluation dataset JSON (default: tests/eval/eval_dataset.json)"
            echo "  --provider PROVIDER   LLM provider: claude, openai, fallback (default: fallback)"
            echo "  --category CATEGORY   Filter to specific category (straightforward, multi_chunk, etc.)"
            echo "  --k N                 Number of documents to retrieve (default: 10)"
            echo "  --rerank-top-k N      Documents to keep after reranking (default: 5)"
            echo "  --output DIR          Output directory for results (default: eval_results/)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date -u +"%Y%m%d-%H%M%S")
RUN_ID="eval-${TIMESTAMP}"

echo "============================================================"
echo " RAG Pipeline Evaluation"
echo "============================================================"
echo " Run ID:      $RUN_ID"
echo " Dataset:     $DATASET"
echo " Provider:    $PROVIDER"
echo " K:           $K"
echo " Rerank Top K: $RERANK_TOP_K"
echo " Category:    ${CATEGORY:-all}"
echo " Output:      $OUTPUT_DIR"
echo "============================================================"
echo ""

# Run the evaluation via Python
python3 -c "
import json, sys
from pathlib import Path

from src.evaluation.dataset import EvalDataset
from src.evaluation.metrics import RAGMetrics
from src.evaluation.runner import EvalRunner
from src.generation.llm import LLMFactory

# Load dataset
dataset = EvalDataset.load('${DATASET}')
print(f'Loaded {len(dataset)} Q&A pairs')

category = '${CATEGORY}'
if category:
    dataset = dataset.filter_by_category(category)
    print(f'Filtered to {len(dataset)} pairs (category: {category})')

if len(dataset) == 0:
    print('ERROR: No Q&A pairs to evaluate')
    sys.exit(1)

# Set up LLM evaluator
llm = LLMFactory.create(provider='${PROVIDER}')
metrics = RAGMetrics(llm=llm)

# Set up the RAG chain
from src.retrieval.retriever import HybridRetriever, SemanticRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker
from src.vectorstore.chroma_store import ChromaVectorStore
from src.generation.chain import RAGChain

store = ChromaVectorStore()
semantic = SemanticRetriever(store)
bm25 = BM25Retriever()
retriever = HybridRetriever(semantic=semantic, bm25=bm25)
reranker = CrossEncoderReranker()
chain = RAGChain(retriever=retriever, llm=llm, reranker=reranker)

# Run evaluation
runner = EvalRunner(metrics=metrics, k=${K}, rerank_top_k=${RERANK_TOP_K})
report = runner.run(dataset, chain, run_id='${RUN_ID}')

# Save results
output_dir = Path('${OUTPUT_DIR}')
report.save_json(output_dir / f'${RUN_ID}.json')
report.save_markdown(output_dir / f'${RUN_ID}.md')

# Print summary table
print()
print('============================================================')
print(' EVALUATION RESULTS')
print('============================================================')
agg = report.aggregate.to_dict()
for metric_name in ('faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'):
    stats = agg.get(metric_name, {})
    if stats:
        mean = stats.get('mean', 0)
        std = stats.get('std', 0)
        print(f'  {metric_name:25s}  {mean:.4f} (+/- {std:.4f})')
    else:
        print(f'  {metric_name:25s}  N/A')
print()
print(f'  Average latency:          {report.avg_latency_ms:.1f} ms')
print(f'  Total latency:            {report.total_latency_ms:.1f} ms')
print(f'  Questions evaluated:      {report.total_questions}')
print('============================================================')

# Output for CI: write scores to a simple file for threshold checks
scores_file = output_dir / 'latest_scores.json'
scores_file.write_text(json.dumps(agg, indent=2))
print(f'Scores saved to {scores_file}')
"

echo ""
echo "Results saved to: $OUTPUT_DIR/$RUN_ID.json"
echo "Markdown report:  $OUTPUT_DIR/$RUN_ID.md"
echo "Done."
