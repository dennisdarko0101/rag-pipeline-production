"""Document model used across the RAG pipeline."""

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document or chunk flowing through the pipeline.

    Used as the universal data structure from ingestion through retrieval.
    Loaders produce Documents, chunkers split them into smaller Documents,
    and the retrieval layer returns ranked Documents.
    """

    doc_id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def source(self) -> str:
        """Shortcut to metadata['source']."""
        return str(self.metadata.get("source", ""))

    @property
    def source_label(self) -> str:
        """Display-friendly source name.

        Returns the bare filename for local file paths (so citations and source
        cards show "rag_systems.md" rather than an absolute path), and leaves web
        URLs intact. Splits on both "/" and "\\" so it is correct regardless of
        the OS the documents were ingested on.
        """
        src = self.source
        if src.startswith(("http://", "https://")):
            return src
        return re.split(r"[\\/]", src)[-1] or src

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def fingerprint(self) -> str | None:
        """Return the document fingerprint if set by preprocessor."""
        return self.metadata.get("fingerprint")
