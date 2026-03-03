#!/usr/bin/env bash
set -euo pipefail

echo "Setting up RAG Pipeline Production..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.11"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.11+ required, found $python_version"
    exit 1
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate
echo "Installing dependencies..."
pip install -e ".[dev]"

# Setup environment file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example - please add your API keys"
fi

# Create data directories
mkdir -p data/chroma data/sample_docs

echo "Setup complete! Activate the environment with: source .venv/bin/activate"
