"""Tests for document loaders."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.ingestion.loader import (
    DocumentLoader,
    MarkdownLoader,
    TextLoader,
    WebLoader,
    get_loader,
)
from src.models.document import Document

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_docs"


# --- MarkdownLoader ---


class TestMarkdownLoader:
    def test_load_markdown_file(self) -> None:
        loader = MarkdownLoader()
        docs = loader.load(str(SAMPLE_DIR / "ai_agents.md"))

        assert len(docs) == 1
        doc = docs[0]
        assert isinstance(doc, Document)
        assert "AI Agent" in doc.content
        assert doc.metadata["file_type"] == "markdown"
        assert doc.metadata["title"] == "AI Agent Architectures: Patterns and Design Principles"
        assert doc.metadata["source"].endswith("ai_agents.md")
        assert "timestamp" in doc.metadata

    def test_load_all_sample_docs(self) -> None:
        loader = MarkdownLoader()
        for md_file in SAMPLE_DIR.glob("*.md"):
            docs = loader.load(str(md_file))
            assert len(docs) == 1
            assert docs[0].content.strip() != ""
            assert docs[0].metadata["file_type"] == "markdown"

    def test_missing_file_raises(self) -> None:
        loader = MarkdownLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.md")


# --- TextLoader ---


class TestTextLoader:
    def test_load_text_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "sample.txt"
        text_file.write_text("Hello, this is a test document.\nWith two lines.", encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(str(text_file))

        assert len(docs) == 1
        assert docs[0].content == "Hello, this is a test document.\nWith two lines."
        assert docs[0].metadata["file_type"] == "text"
        assert docs[0].metadata["source"].endswith("sample.txt")

    def test_load_empty_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "empty.txt"
        text_file.write_text("", encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(str(text_file))
        assert len(docs) == 1
        assert docs[0].content == ""

    def test_missing_file_raises(self) -> None:
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.txt")


# --- WebLoader (mocked) ---


class TestWebLoader:
    @patch("src.ingestion.loader.httpx.get")
    def test_load_web_page(self, mock_get: MagicMock) -> None:
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Navigation links</nav>
            <main><p>This is the main content of the page.</p></main>
            <footer>Footer info</footer>
        </body>
        </html>
        """
        mock_response = MagicMock()
        mock_response.text = html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        loader = WebLoader()
        docs = loader.load("https://example.com/article")

        assert len(docs) == 1
        doc = docs[0]
        assert "main content" in doc.content
        # nav and footer should be stripped
        assert "Navigation links" not in doc.content
        assert "Footer info" not in doc.content
        assert doc.metadata["file_type"] == "web"
        assert doc.metadata["title"] == "Test Page"
        assert doc.metadata["source"] == "https://example.com/article"
        assert doc.metadata["status_code"] == 200

    @patch("src.ingestion.loader.httpx.get")
    def test_web_loader_http_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        loader = WebLoader()
        with pytest.raises(httpx.HTTPStatusError):
            loader.load("https://example.com/missing")


# --- get_loader factory ---


class TestGetLoader:
    def test_returns_markdown_loader(self) -> None:
        loader = get_loader("docs/readme.md")
        assert isinstance(loader, MarkdownLoader)

    def test_returns_text_loader(self) -> None:
        loader = get_loader("notes.txt")
        assert isinstance(loader, TextLoader)

    def test_returns_web_loader(self) -> None:
        loader = get_loader("https://example.com/page")
        assert isinstance(loader, WebLoader)

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_loader("data.csv")

    def test_base_class_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            DocumentLoader()  # type: ignore[abstract]
