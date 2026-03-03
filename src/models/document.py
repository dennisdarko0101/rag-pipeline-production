"""Document model used across the RAG pipeline."""

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
        return self.metadata.get("source", "")

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def fingerprint(self) -> str | None:
        """Return the document fingerprint if set by preprocessor."""
        return self.metadata.get("fingerprint")
