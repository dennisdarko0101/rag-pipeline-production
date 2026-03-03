"""Chunking strategies for splitting documents into smaller pieces."""

from abc import ABC, abstractmethod

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.models.document import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseChunker(ABC):
    """Base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks.

        Args:
            documents: List of documents to chunk.

        Returns:
            List of chunked Document objects with preserved metadata.
        """


class RecursiveChunker(BaseChunker):
    """Wraps LangChain RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            "recursive_chunker_init",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []

        for doc in documents:
            if not doc.content.strip():
                continue

            text_chunks = self._splitter.split_text(doc.content)

            for idx, text in enumerate(text_chunks):
                chunk_doc = Document(
                    content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "total_chunks": len(text_chunks),
                        "parent_doc_id": doc.doc_id,
                    },
                )
                chunks.append(chunk_doc)

        logger.info(
            "recursive_chunking_complete",
            input_docs=len(documents),
            output_chunks=len(chunks),
        )
        return chunks


class SemanticChunker(BaseChunker):
    """Groups sentences by embedding similarity using cosine distance.

    Sentences that are semantically similar are kept together. A new chunk
    boundary is created when the similarity between consecutive sentence
    groups drops below the threshold.
    """

    def __init__(
        self,
        embedding_fn: callable | None = None,
        similarity_threshold: float = 0.5,
        max_chunk_size: int | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size or settings.chunk_size
        self._embedding_fn = embedding_fn
        logger.info(
            "semantic_chunker_init",
            threshold=self.similarity_threshold,
            max_chunk_size=self.max_chunk_size,
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on period, question mark, or exclamation."""
        sentences: list[str] = []
        current = ""
        for char in text:
            current += char
            if char in ".!?" and current.strip():
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        return sentences

    def _get_embeddings(self, sentences: list[str]) -> np.ndarray:
        """Get embeddings for a list of sentences."""
        if self._embedding_fn is None:
            raise ValueError(
                "SemanticChunker requires an embedding_fn. "
                "Pass a callable that maps List[str] -> np.ndarray."
            )
        return np.array(self._embedding_fn(sentences))

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []

        for doc in documents:
            if not doc.content.strip():
                continue

            sentences = self._split_sentences(doc.content)
            if not sentences:
                continue

            if len(sentences) == 1:
                chunks.append(
                    Document(
                        content=sentences[0],
                        metadata={**doc.metadata, "chunk_index": 0, "total_chunks": 1,
                                  "parent_doc_id": doc.doc_id},
                    )
                )
                continue

            embeddings = self._get_embeddings(sentences)

            # Group sentences by similarity
            current_group: list[str] = [sentences[0]]
            groups: list[list[str]] = []

            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

                current_text = " ".join(current_group + [sentences[i]])
                if sim >= self.similarity_threshold and len(current_text) <= self.max_chunk_size:
                    current_group.append(sentences[i])
                else:
                    groups.append(current_group)
                    current_group = [sentences[i]]

            if current_group:
                groups.append(current_group)

            for idx, group in enumerate(groups):
                chunk_doc = Document(
                    content=" ".join(group),
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "total_chunks": len(groups),
                        "parent_doc_id": doc.doc_id,
                    },
                )
                chunks.append(chunk_doc)

        logger.info(
            "semantic_chunking_complete",
            input_docs=len(documents),
            output_chunks=len(chunks),
        )
        return chunks


def create_chunker(
    strategy: str = "recursive",
    **kwargs: object,
) -> BaseChunker:
    """Factory function that returns the right chunker based on strategy name.

    Args:
        strategy: One of "recursive" or "semantic".
        **kwargs: Extra arguments passed to the chunker constructor.

    Returns:
        An instance of the requested chunker.

    Raises:
        ValueError: If the strategy is not recognized.
    """
    chunkers: dict[str, type[BaseChunker]] = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
    }

    chunker_cls = chunkers.get(strategy)
    if chunker_cls is None:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Options: {list(chunkers.keys())}")

    return chunker_cls(**kwargs)  # type: ignore[arg-type]
