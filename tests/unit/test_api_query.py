"""Unit tests for the /query endpoint."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


@dataclass
class _FakeSource:
    source_name: str = "doc.md"
    chunk_text: str = "chunk content"
    chunk_index: int = 0
    relevance_score: float = 0.9


@dataclass
class _FakeCitation:
    source: str = "doc.md"
    chunk_index: int = 0
    raw_text: str = "[Source: doc.md, Chunk 0]"


@dataclass
class _FakeRAGResponse:
    answer: str = "Test answer."
    sources: list = field(default_factory=lambda: [_FakeSource()])
    citations: list = field(default_factory=lambda: [_FakeCitation()])
    metadata: dict = field(default_factory=lambda: {"latency_ms": 42.0})


class TestQueryEndpoint:
    """Tests for POST /api/v1/query."""

    @patch("src.api.routes.query._build_rag_chain")
    def test_successful_query(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.return_value = _FakeRAGResponse()
        mock_build.return_value = mock_chain

        response = client.post("/api/v1/query", json={"question": "What is RAG?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_name"] == "doc.md"
        assert len(data["citations"]) == 1
        assert data["metadata"]["latency_ms"] == 42.0

    @patch("src.api.routes.query._build_rag_chain")
    def test_query_passes_parameters(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.return_value = _FakeRAGResponse()
        mock_build.return_value = mock_chain

        client.post(
            "/api/v1/query",
            json={
                "question": "Test?",
                "k": 20,
                "rerank": False,
                "rerank_top_k": 3,
                "provider": "openai",
            },
        )

        mock_build.assert_called_once_with("openai", False)
        mock_chain.query.assert_called_once_with(question="Test?", k=20, rerank_top_k=3)

    def test_query_empty_question_rejected(self) -> None:
        response = client.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_missing_question_rejected(self) -> None:
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_invalid_provider_rejected(self) -> None:
        response = client.post("/api/v1/query", json={"question": "Hi", "provider": "gemini"})
        assert response.status_code == 422

    def test_query_k_out_of_range_rejected(self) -> None:
        response = client.post("/api/v1/query", json={"question": "Hi", "k": 0})
        assert response.status_code == 422

    @patch("src.api.routes.query._build_rag_chain")
    def test_query_pipeline_error_returns_500(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.side_effect = RuntimeError("LLM exploded")
        mock_build.return_value = mock_chain

        response = client.post("/api/v1/query", json={"question": "Hello?"})

        assert response.status_code == 500
        data = response.json()
        assert "internal error" in data["detail"].lower()

    @patch("src.api.routes.query._build_rag_chain")
    def test_query_value_error_returns_400(self, mock_build: MagicMock) -> None:
        mock_build.side_effect = ValueError("Bad provider")

        response = client.post("/api/v1/query", json={"question": "Hello?"})

        assert response.status_code == 400

    @patch("src.api.routes.query._build_rag_chain")
    def test_query_no_sources(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.return_value = _FakeRAGResponse(
            answer="No info.", sources=[], citations=[], metadata={}
        )
        mock_build.return_value = mock_chain

        response = client.post("/api/v1/query", json={"question": "Unknown?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "No info."
        assert data["sources"] == []

    @patch("src.api.routes.query._build_rag_chain")
    def test_query_default_parameters(self, mock_build: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.query.return_value = _FakeRAGResponse()
        mock_build.return_value = mock_chain

        client.post("/api/v1/query", json={"question": "Defaults?"})

        mock_build.assert_called_once_with("fallback", True)
        mock_chain.query.assert_called_once_with(question="Defaults?", k=10, rerank_top_k=5)

    def test_query_response_has_rate_limit_headers(self) -> None:
        """Rate limit headers should be present (from middleware)."""
        with patch("src.api.routes.query._build_rag_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.query.return_value = _FakeRAGResponse()
            mock_build.return_value = mock_chain

            response = client.post("/api/v1/query", json={"question": "Hi"})

        assert "x-ratelimit-limit" in response.headers
        assert "x-ratelimit-remaining" in response.headers
