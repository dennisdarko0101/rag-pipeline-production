"""Unit tests for the /health endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

_CHROMA_PATCH = "src.vectorstore.chroma_store.ChromaVectorStore"


class TestHealthEndpoint:
    """Tests for GET /health."""

    @patch(_CHROMA_PATCH)
    def test_health_healthy(self, mock_store_cls) -> None:  # noqa: ANN001
        mock_store = mock_store_cls.return_value
        mock_store.get_stats.return_value = {"total_documents": 42}

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["components"]["vectorstore"]["status"] == "healthy"
        assert "42" in data["components"]["vectorstore"]["details"]

    @patch(_CHROMA_PATCH)
    def test_health_degraded_when_vectorstore_fails(self, mock_store_cls) -> None:  # noqa: ANN001
        mock_store_cls.side_effect = Exception("Connection refused")

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["vectorstore"]["status"] == "unhealthy"

    def test_health_returns_version(self) -> None:
        with patch(_CHROMA_PATCH) as mock_cls:
            mock_cls.return_value.get_stats.return_value = {"total_documents": 0}
            response = client.get("/health")

        assert response.json()["version"] == "0.1.0"

    def test_health_not_rate_limited(self) -> None:
        """Health endpoint should bypass rate limiting."""
        with patch(_CHROMA_PATCH) as mock_cls:
            mock_cls.return_value.get_stats.return_value = {"total_documents": 0}
            # Make many requests — should not get 429
            for _ in range(100):
                response = client.get("/health")
                assert response.status_code == 200
