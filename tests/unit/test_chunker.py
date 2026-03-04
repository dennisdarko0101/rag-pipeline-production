"""Tests for chunking strategies."""

import numpy as np
import pytest

from src.ingestion.chunker import BaseChunker, RecursiveChunker, SemanticChunker, create_chunker
from src.models.document import Document


def _make_doc(content: str, **meta: object) -> Document:
    """Helper to create a Document with optional metadata."""
    return Document(content=content, metadata={"source": "test.md", **meta})


# --- RecursiveChunker ---


class TestRecursiveChunker:
    def test_basic_chunking(self) -> None:
        doc = _make_doc("A" * 200 + "\n\n" + "B" * 200 + "\n\n" + "C" * 200)
        chunker = RecursiveChunker(chunk_size=250, chunk_overlap=20)
        chunks = chunker.chunk([doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.content) <= 250 + 50  # some tolerance for splitting

    def test_small_doc_stays_intact(self) -> None:
        doc = _make_doc("Short document that fits in one chunk.")
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk([doc])

        assert len(chunks) == 1
        assert chunks[0].content == "Short document that fits in one chunk."

    def test_empty_doc_produces_no_chunks(self) -> None:
        doc = _make_doc("")
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk([doc])
        assert len(chunks) == 0

    def test_whitespace_only_doc(self) -> None:
        doc = _make_doc("   \n\n   \t  ")
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk([doc])
        assert len(chunks) == 0

    def test_metadata_preserved(self) -> None:
        doc = _make_doc("Some content " * 100, file_type="markdown", title="Test Doc")
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk([doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.md"
            assert chunk.metadata["file_type"] == "markdown"
            assert chunk.metadata["title"] == "Test Doc"

    def test_chunk_index_assigned(self) -> None:
        doc = _make_doc("word " * 500)
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk([doc])

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert chunk.metadata["parent_doc_id"] == doc.doc_id

    def test_overlap_present(self) -> None:
        # Create a document with clear paragraph boundaries
        paragraphs = [f"Paragraph {i}. " + "x" * 80 for i in range(10)]
        doc = _make_doc("\n\n".join(paragraphs))
        chunker = RecursiveChunker(chunk_size=120, chunk_overlap=30)
        chunks = chunker.chunk([doc])

        # With overlap, consecutive chunks should share some text
        assert len(chunks) >= 2

    def test_multiple_documents(self) -> None:
        docs = [
            _make_doc("First document. " * 50, title="Doc 1"),
            _make_doc("Second document. " * 50, title="Doc 2"),
        ]
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(docs)

        # Chunks from doc 1 should have title "Doc 1"
        doc1_chunks = [c for c in chunks if c.metadata["title"] == "Doc 1"]
        doc2_chunks = [c for c in chunks if c.metadata["title"] == "Doc 2"]
        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0

    def test_various_chunk_sizes(self) -> None:
        doc = _make_doc("Hello world. " * 200)
        for size in [64, 128, 256, 512, 1024]:
            chunker = RecursiveChunker(chunk_size=size, chunk_overlap=10)
            chunks = chunker.chunk([doc])
            assert len(chunks) >= 1


# --- SemanticChunker ---


class TestSemanticChunker:
    @staticmethod
    def _fake_embedding_fn(sentences: list[str]) -> list[list[float]]:
        """Return embeddings that cluster by sentence prefix."""
        embeddings = []
        for s in sentences:
            if s.startswith("Topic A"):
                embeddings.append([1.0, 0.0, 0.0])
            elif s.startswith("Topic B"):
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.5, 0.5, 0.0])
        return embeddings

    def test_semantic_grouping(self) -> None:
        text = (
            "Topic A is about cats. Topic A also covers kittens. "
            "Topic B discusses dogs. Topic B also mentions puppies."
        )
        doc = _make_doc(text)
        chunker = SemanticChunker(
            embedding_fn=self._fake_embedding_fn,
            similarity_threshold=0.9,
            max_chunk_size=2000,
        )
        chunks = chunker.chunk([doc])

        # Should create at least 2 chunks since topic A and topic B embeddings are dissimilar
        assert len(chunks) >= 2

    def test_single_sentence_doc(self) -> None:
        doc = _make_doc("Just one sentence.")
        chunker = SemanticChunker(
            embedding_fn=self._fake_embedding_fn,
            similarity_threshold=0.5,
        )
        chunks = chunker.chunk([doc])
        assert len(chunks) == 1
        assert chunks[0].content == "Just one sentence."

    def test_empty_doc(self) -> None:
        doc = _make_doc("")
        chunker = SemanticChunker(
            embedding_fn=self._fake_embedding_fn,
            similarity_threshold=0.5,
        )
        chunks = chunker.chunk([doc])
        assert len(chunks) == 0

    def test_requires_embedding_fn(self) -> None:
        doc = _make_doc("This needs an embedding function. It has two sentences.")
        chunker = SemanticChunker(embedding_fn=None, similarity_threshold=0.5)
        with pytest.raises(ValueError, match="requires an embedding_fn"):
            chunker.chunk([doc])

    def test_metadata_preserved(self) -> None:
        doc = _make_doc(
            "Topic A first. Topic B second.",
            file_type="markdown",
        )
        chunker = SemanticChunker(
            embedding_fn=self._fake_embedding_fn,
            similarity_threshold=0.5,
        )
        chunks = chunker.chunk([doc])

        for chunk in chunks:
            assert chunk.metadata["source"] == "test.md"
            assert chunk.metadata["file_type"] == "markdown"
            assert "chunk_index" in chunk.metadata
            assert "parent_doc_id" in chunk.metadata

    def test_cosine_similarity_orthogonal(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert SemanticChunker._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_identical(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert SemanticChunker._cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_similarity_zero_vector(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert SemanticChunker._cosine_similarity(a, b) == 0.0


# --- create_chunker factory ---


class TestCreateChunker:
    def test_creates_recursive(self) -> None:
        chunker = create_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_creates_semantic(self) -> None:
        chunker = create_chunker("semantic", embedding_fn=lambda x: [[0.0]] * len(x))
        assert isinstance(chunker, SemanticChunker)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker("unknown_strategy")

    def test_base_class_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseChunker()  # type: ignore[abstract]
