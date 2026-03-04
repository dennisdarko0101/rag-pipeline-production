"""Document loaders for various file formats and sources."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from src.models.document import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader(ABC):
    """Base class for all document loaders."""

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load documents from a source.

        Args:
            source: File path or URL to load from.

        Returns:
            List of Document objects with content and metadata.
        """


class PDFLoader(DocumentLoader):
    """Extract text from PDF files using PyPDF2."""

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {source}")

        logger.info("loading_pdf", source=source)
        documents: list[Document] = []

        try:
            reader = PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    logger.debug("skipping_empty_page", source=source, page=page_num)
                    continue

                documents.append(
                    Document(
                        content=text,
                        metadata={
                            "source": str(path.resolve()),
                            "file_type": "pdf",
                            "page_number": page_num,
                            "total_pages": len(reader.pages),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
        except Exception as e:
            logger.error("pdf_load_failed", source=source, error=str(e))
            raise

        logger.info("pdf_loaded", source=source, pages=len(documents))
        return documents


class MarkdownLoader(DocumentLoader):
    """Parse Markdown files preserving structure."""

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {source}")

        logger.info("loading_markdown", source=source)

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("markdown_load_failed", source=source, error=str(e))
            raise

        # Extract title from first H1 if present
        title = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("##"):
                title = stripped.removeprefix("# ").strip()
                break

        document = Document(
            content=text,
            metadata={
                "source": str(path.resolve()),
                "file_type": "markdown",
                "title": title,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info("markdown_loaded", source=source, chars=len(text))
        return [document]


class TextLoader(DocumentLoader):
    """Load plain text files."""

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {source}")

        logger.info("loading_text", source=source)

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("text_load_failed", source=source, error=str(e))
            raise

        document = Document(
            content=text,
            metadata={
                "source": str(path.resolve()),
                "file_type": "text",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info("text_loaded", source=source, chars=len(text))
        return [document]


class WebLoader(DocumentLoader):
    """Scrape web pages using httpx + BeautifulSoup."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def load(self, source: str) -> list[Document]:
        logger.info("loading_web", url=source)

        try:
            response = httpx.get(source, timeout=self.timeout, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("web_load_failed", url=source, error=str(e))
            raise

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        document = Document(
            content=text,
            metadata={
                "source": source,
                "file_type": "web",
                "title": title,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status_code": response.status_code,
            },
        )

        logger.info("web_loaded", url=source, chars=len(text))
        return [document]


def get_loader(source: str) -> DocumentLoader:
    """Return the appropriate loader based on the source path or URL.

    Args:
        source: File path or URL.

    Returns:
        An instance of the appropriate DocumentLoader subclass.
    """
    if source.startswith(("http://", "https://")):
        return WebLoader()

    path = Path(source)
    suffix = path.suffix.lower()
    loaders: dict[str, DocumentLoader] = {
        ".pdf": PDFLoader(),
        ".md": MarkdownLoader(),
        ".markdown": MarkdownLoader(),
        ".txt": TextLoader(),
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(loaders.keys())}")

    return loader
