#!/usr/bin/env bash
set -euo pipefail

echo "Running RAG evaluation suite..."
python3 -m pytest tests/eval/ -v --tb=long
echo "Evaluation complete."
