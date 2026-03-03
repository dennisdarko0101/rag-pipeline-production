"""ChromaDB vector store implementation."""

from __future__ import annotations

from typing import Any

from src.config.settings import settings
from src.models.document import Document
from src.utils.logger import get_logger
from src.vectorstore.base import SearchResult, VectorStore

logger = get_logger(__name__)

# Map friendly names to ChromaDB distance functions
_DISTANCE_FUNCTIONS: dict[str, str] = {
    "cosine": "cosine",
    "l2": "l2",
    "ip": "ip",
}


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store with persistent storage.

    Supports metadata filtering, batch upsert, and configurable
    distance metrics.
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_dir: str | None = None,
        distance_metric: str = "cosine",
    ) -> None:
        import chromadb as _chromadb

        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._collection_name = collection_name
        self._distance_fn = _DISTANCE_FUNCTIONS.get(distance_metric)
        if self._distance_fn is None:
            raise ValueError(
                f"Unknown distance metric: {distance_metric}. "
                f"Options: {list(_DISTANCE_FUNCTIONS.keys())}"
            )

        self._client = _chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance_fn},
        )

        logger.info(
            "chroma_store_init",
            collection=self._collection_name,
            persist_dir=self._persist_dir,
            distance_metric=distance_metric,
        )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Add documents with embeddings to ChromaDB.

        Uses upsert to handle duplicate IDs gracefully.

        Args:
            documents: List of documents to store.
            embeddings: Corresponding embedding vectors.

        Returns:
            List of document IDs that were stored.
        """
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(documents)} documents but {len(embeddings)} embeddings"
            )
        if not documents:
            return []

        ids = [doc.doc_id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [self._sanitize_metadata(doc.metadata) for doc in documents]

        # Batch upsert in chunks of 500 (ChromaDB recommendation)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=contents[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(
            "documents_added",
            count=len(documents),
            collection=self._collection_name,
        )
        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents in ChromaDB.

        Args:
            query_embedding: The query embedding vector.
            k: Number of results to return.
            where: Optional metadata filter (ChromaDB where clause).

        Returns:
            List of SearchResult objects ranked by similarity.
        """
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = self._collection.query(**query_params)

        search_results: list[SearchResult] = []
        if results["ids"] and results["ids"][0]:
            for rank, (doc_id, content, metadata, distance) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    strict=True,
                )
            ):
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1.0 - distance if self._distance_fn == "cosine" else -distance

                doc = Document(
                    doc_id=doc_id,
                    content=content or "",
                    metadata=metadata or {},
                )
                search_results.append(SearchResult(document=doc, score=score, rank=rank))

        logger.info(
            "search_complete",
            results=len(search_results),
            k=k,
            has_filter=where is not None,
        )
        return search_results

    def delete(self, doc_ids: list[str]) -> int:
        """Delete documents by their IDs.

        Args:
            doc_ids: List of document IDs to delete.

        Returns:
            Number of documents deleted.
        """
        if not doc_ids:
            return 0

        # Check which IDs actually exist
        existing = self._collection.get(ids=doc_ids)
        existing_ids = existing["ids"]

        if existing_ids:
            self._collection.delete(ids=existing_ids)

        logger.info("documents_deleted", count=len(existing_ids), requested=len(doc_ids))
        return len(existing_ids)

    def get_stats(self) -> dict:
        """Get collection statistics.

        Returns:
            Dictionary with total_documents, collection_name, persist_dir.
        """
        count = self._collection.count()
        return {
            "total_documents": count,
            "collection_name": self._collection_name,
            "persist_dir": self._persist_dir,
            "distance_metric": self._distance_fn,
        }

    def list_collections(self) -> list[str]:
        """List all collection names in this ChromaDB instance.

        Returns:
            List of collection name strings.
        """
        return [c.name for c in self._client.list_collections()]

    def delete_collection(self, name: str | None = None) -> None:
        """Delete a collection.

        Args:
            name: Collection name to delete. Defaults to current collection.
        """
        target = name or self._collection_name
        self._client.delete_collection(target)
        logger.info("collection_deleted", collection=target)

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize metadata for ChromaDB compatibility.

        ChromaDB only supports str, int, float, and bool values in metadata.
        Lists, dicts, and None values are converted or removed.

        Args:
            metadata: Raw metadata dictionary.

        Returns:
            Sanitized metadata dictionary.
        """
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[key] = ", ".join(str(v) for v in value)
            elif value is not None:
                sanitized[key] = str(value)
            # Skip None values
        return sanitized
