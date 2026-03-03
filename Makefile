.PHONY: install test lint format run docker-build docker-up seed eval clean

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	streamlit run ui/app.py

docker-build:
	docker build -f docker/Dockerfile -t rag-pipeline .

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

seed:
	bash scripts/seed_db.sh

eval:
	bash scripts/run_eval.sh

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
