"""Unit tests for the /ingest endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app, raise_server_exceptions=False)


class TestIngestEndpoint:
    """Tests for POST /api/v1/ingest."""

    @patch("src.api.routes.ingest._run_ingestion")
    def test_successful_ingest(self, mock_ingest) -> None:  # noqa: ANN001
        mock_ingest.return_value = (1, 12)

        response = client.post(
            "/api/v1/ingest",
            json={"source_path": "data/sample_docs/rag_systems.md"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["documents_processed"] == 1
        assert data["chunks_created"] == 12
        assert "Successfully" in data["message"]

    @patch("src.api.routes.ingest._run_ingestion")
    def test_ingest_with_doc_type(self, mock_ingest) -> None:  # noqa: ANN001
        mock_ingest.return_value = (1, 8)

        response = client.post(
            "/api/v1/ingest",
            json={"source_path": "doc.md", "doc_type": "markdown"},
        )

        assert response.status_code == 200
        mock_ingest.assert_called_once_with("doc.md", "markdown")

    def test_ingest_empty_source_rejected(self) -> None:
        response = client.post("/api/v1/ingest", json={"source_path": ""})
        assert response.status_code == 422

    def test_ingest_missing_source_rejected(self) -> None:
        response = client.post("/api/v1/ingest", json={})
        assert response.status_code == 422

    def test_ingest_invalid_doc_type_rejected(self) -> None:
        response = client.post(
            "/api/v1/ingest",
            json={"source_path": "file.txt", "doc_type": "xlsx"},
        )
        assert response.status_code == 422

    @patch("src.api.routes.ingest._run_ingestion")
    def test_ingest_file_not_found_returns_400(self, mock_ingest) -> None:  # noqa: ANN001
        mock_ingest.side_effect = FileNotFoundError("not found")

        response = client.post("/api/v1/ingest", json={"source_path": "/no/such/file.md"})

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    @patch("src.api.routes.ingest._run_ingestion")
    def test_ingest_value_error_returns_400(self, mock_ingest) -> None:  # noqa: ANN001
        mock_ingest.side_effect = ValueError("Unsupported format")

        response = client.post("/api/v1/ingest", json={"source_path": "file.xyz"})

        assert response.status_code == 400

    @patch("src.api.routes.ingest._run_ingestion")
    def test_ingest_internal_error_returns_500(self, mock_ingest) -> None:  # noqa: ANN001
        mock_ingest.side_effect = RuntimeError("DB crash")

        response = client.post("/api/v1/ingest", json={"source_path": "file.md"})

        assert response.status_code == 500

    @patch("src.api.routes.ingest._run_ingestion")
    def test_ingest_default_doc_type(self, mock_ingest) -> None:  # noqa: ANN001
        mock_ingest.return_value = (1, 5)

        client.post("/api/v1/ingest", json={"source_path": "file.md"})

        mock_ingest.assert_called_once_with("file.md", "auto")


class TestIngestUploadEndpoint:
    """Tests for POST /api/v1/ingest/upload."""

    @patch("src.api.routes.ingest._run_ingestion")
    def test_upload_file(self, mock_ingest) -> None:  # noqa: ANN001
        mock_ingest.return_value = (1, 6)

        response = client.post(
            "/api/v1/ingest/upload",
            files={"file": ("test.md", b"# Hello\n\nWorld", "text/markdown")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["documents_processed"] == 1
        assert data["chunks_created"] == 6
        assert "test.md" in data["message"]

    @patch("src.api.routes.ingest._run_ingestion")
    def test_upload_ingestion_error(self, mock_ingest) -> None:  # noqa: ANN001
        mock_ingest.side_effect = RuntimeError("fail")

        response = client.post(
            "/api/v1/ingest/upload",
            files={"file": ("test.txt", b"content", "text/plain")},
        )

        assert response.status_code == 500
