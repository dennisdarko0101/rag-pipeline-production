# Contributing

Thanks for your interest in contributing to the RAG Pipeline project!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/dennisdarko/rag-pipeline-production.git
cd rag-pipeline-production

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Code Quality

All code must pass these checks before merging:

```bash
# Lint
ruff check src/ tests/ ui/

# Format
ruff format src/ tests/ ui/

# Type check
mypy src/

# Tests (272 passing, 80%+ coverage required)
pytest tests/ -v --cov=src --cov-fail-under=80
```

Or use the Makefile shortcuts:

```bash
make lint       # ruff check + mypy
make format     # ruff fix + format
make test       # pytest
make test-cov   # pytest with coverage
```

## Making Changes

1. **Create a branch** from `main`
2. **Write tests** for any new functionality
3. **Run the full check suite** before submitting a PR
4. **Keep commits focused** -- one logical change per commit
5. **Write clear commit messages** that explain *why*, not just *what*

## Project Structure

- `src/` -- Core pipeline code (ingestion, embeddings, retrieval, generation, evaluation, API)
- `tests/` -- Unit and integration tests (all external APIs mocked)
- `ui/` -- Streamlit dashboard (communicates with backend via HTTP only)
- `docker/` -- Dockerfiles and compose configuration
- `docs/` -- Architecture, deployment, evaluation, and handoff documentation

## Adding Tests

Tests live in `tests/unit/` and `tests/integration/`. All external APIs (OpenAI, Anthropic, ChromaDB) are mocked -- no API keys needed to run the test suite.

```bash
# Run a specific test file
pytest tests/unit/test_chain.py -v

# Run tests matching a pattern
pytest tests/ -k "test_query" -v
```

## Reporting Issues

Open an issue on GitHub with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
