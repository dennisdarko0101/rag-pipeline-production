"""Integration tests for the FastAPI application.

Tests the full request/response cycle through the ASGI stack,
including middleware, error handling, and route registration.
"""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestAppConfiguration:
    """Test that the FastAPI app is configured correctly."""

    def test_openapi_schema_available(self) -> None:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "RAG Pipeline API"
        assert schema["info"]["version"] == "0.1.0"

    def test_docs_available(self) -> None:
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self) -> None:
        response = client.get("/redoc")
        assert response.status_code == 200


class TestRouteRegistration:
    """Test all routes are registered under correct prefixes."""

    def test_health_at_root(self) -> None:
        with patch("src.vectorstore.chroma_store.ChromaVectorStore") as mock_cls:
            mock_cls.return_value.get_stats.return_value = {"total_documents": 0}
            response = client.get("/health")
        assert response.status_code == 200

    def test_query_at_api_v1(self) -> None:
        # Without mocking the chain, we just test the route exists (422 = validation error)
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422  # Missing required field

    def test_ingest_at_api_v1(self) -> None:
        response = client.post("/api/v1/ingest", json={})
        assert response.status_code == 422

    def test_evaluate_at_api_v1(self) -> None:
        response = client.post("/api/v1/evaluate", json={})
        assert response.status_code == 422


class TestMiddleware:
    """Test middleware integration."""

    def test_request_id_header_returned(self) -> None:
        with patch("src.vectorstore.chroma_store.ChromaVectorStore") as mock_cls:
            mock_cls.return_value.get_stats.return_value = {"total_documents": 0}
            response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_custom_request_id_preserved(self) -> None:
        with patch("src.vectorstore.chroma_store.ChromaVectorStore") as mock_cls:
            mock_cls.return_value.get_stats.return_value = {"total_documents": 0}
            response = client.get("/health", headers={"x-request-id": "my-trace-123"})
        assert response.headers["x-request-id"] == "my-trace-123"

    def test_rate_limit_headers_on_api_routes(self) -> None:
        with patch("src.api.routes.query._build_rag_chain") as mock_build:
            mock_chain = MagicMock()
            mock_chain.query.return_value = MagicMock(
                answer="ok",
                sources=[],
                citations=[],
                metadata={},
            )
            mock_build.return_value = mock_chain
            response = client.post("/api/v1/query", json={"question": "hi"})

        assert "x-ratelimit-limit" in response.headers


class TestCORS:
    """Test CORS headers."""

    def test_cors_headers_on_options(self) -> None:
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should respond (200 or allow the preflight)
        assert response.status_code == 200


class TestGlobalErrorHandler:
    """Test the global exception handler."""

    @patch("src.api.routes.query._build_rag_chain")
    def test_unhandled_exception_returns_500(self, mock_build: MagicMock) -> None:
        mock_build.side_effect = TypeError("unexpected")

        response = client.post("/api/v1/query", json={"question": "test?"})

        assert response.status_code == 500
