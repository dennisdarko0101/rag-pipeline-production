"""Abstract vector store interface and shared types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.models.document import Document


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    document: Document
    score: float
    rank: int


class VectorStore(ABC):
    """Abstract interface for vector store implementations."""

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Add documents with their embeddings to the store.

        Args:
            documents: List of documents to store.
            embeddings: Corresponding embedding vectors.

        Returns:
            List of document IDs that were stored.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: The query embedding vector.
            k: Number of results to return.
            where: Optional metadata filter.

        Returns:
            List of SearchResult objects ranked by similarity.
        """

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> int:
        """Delete documents by their IDs.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents actually deleted.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """Get statistics about the vector store.

        Returns:
            Dictionary with total_documents, collection_name, etc.
        """
