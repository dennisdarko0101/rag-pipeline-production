"""Unit tests for the /evaluate endpoint."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


@dataclass
class _FakeRAGResponse:
    answer: str = "Generated answer."
    sources: list = field(default_factory=lambda: [MagicMock()])
    citations: list = field(default_factory=list)
    metadata: dict = field(default_factory=lambda: {"latency_ms": 50.0})


class TestEvaluateEndpoint:
    """Tests for POST /api/v1/evaluate."""

    @patch("src.api.routes.query._build_rag_chain")
    def test_successful_evaluation(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.return_value = _FakeRAGResponse()
        mock_build.return_value = mock_chain

        response = client.post(
            "/api/v1/evaluate",
            json={
                "qa_pairs": [
                    {"question": "What is RAG?", "ground_truth": "RAG stands for..."},
                    {"question": "How does BM25 work?", "ground_truth": "BM25 is..."},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["eval_id"]  # non-empty UUID
        assert data["result"]["total_questions"] == 2
        assert len(data["result"]["results"]) == 2
        assert data["result"]["results"][0]["question"] == "What is RAG?"
        assert data["result"]["results"][0]["answer"] == "Generated answer."

    @patch("src.api.routes.query._build_rag_chain")
    def test_evaluation_single_pair(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.return_value = _FakeRAGResponse()
        mock_build.return_value = mock_chain

        response = client.post(
            "/api/v1/evaluate",
            json={
                "qa_pairs": [
                    {"question": "Q1?", "ground_truth": "A1"},
                ],
            },
        )

        assert response.status_code == 200
        assert response.json()["result"]["total_questions"] == 1

    @patch("src.api.routes.query._build_rag_chain")
    def test_evaluation_query_failure_handled(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.side_effect = RuntimeError("LLM error")
        mock_build.return_value = mock_chain

        response = client.post(
            "/api/v1/evaluate",
            json={
                "qa_pairs": [
                    {"question": "Q1?", "ground_truth": "A1"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["results"][0]["answer"] == "[ERROR]"
        assert data["result"]["results"][0]["num_sources"] == 0

    @patch("src.api.routes.query._build_rag_chain")
    def test_evaluation_chain_init_failure(self, mock_build: MagicMock) -> None:
        mock_build.side_effect = RuntimeError("No API key")

        response = client.post(
            "/api/v1/evaluate",
            json={
                "qa_pairs": [
                    {"question": "Q1?", "ground_truth": "A1"},
                ],
            },
        )

        assert response.status_code == 500

    def test_evaluation_empty_pairs_rejected(self) -> None:
        response = client.post("/api/v1/evaluate", json={"qa_pairs": []})
        assert response.status_code == 422

    def test_evaluation_missing_ground_truth_rejected(self) -> None:
        response = client.post(
            "/api/v1/evaluate",
            json={"qa_pairs": [{"question": "Q?"}]},
        )
        assert response.status_code == 422

    @patch("src.api.routes.query._build_rag_chain")
    def test_evaluation_avg_latency(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.return_value = _FakeRAGResponse()
        mock_build.return_value = mock_chain

        response = client.post(
            "/api/v1/evaluate",
            json={
                "qa_pairs": [
                    {"question": "Q1?", "ground_truth": "A1"},
                    {"question": "Q2?", "ground_truth": "A2"},
                ],
            },
        )

        data = response.json()
        assert data["result"]["avg_latency_ms"] >= 0

    @patch("src.api.routes.query._build_rag_chain")
    def test_evaluation_custom_provider(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.return_value = _FakeRAGResponse()
        mock_build.return_value = mock_chain

        client.post(
            "/api/v1/evaluate",
            json={
                "qa_pairs": [{"question": "Q?", "ground_truth": "A"}],
                "provider": "openai",
                "k": 5,
                "rerank": False,
            },
        )

        mock_build.assert_called_once_with("openai", False)
