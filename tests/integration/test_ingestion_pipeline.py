"""Integration test: load → chunk → embed (mocked) → store → search."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.ingestion.chunker import RecursiveChunker
from src.ingestion.loader import MarkdownLoader
from src.ingestion.preprocessor import PreprocessingPipeline

if TYPE_CHECKING:
    from src.models.document import Document

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_docs"


def _fake_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate deterministic fake embeddings based on text length."""
    embeddings = []
    for text in texts:
        char_sum = sum(ord(c) for c in text[:50])
        base = (char_sum % 100) / 100.0
        embeddings.append([base + (i % 10) * 0.01 for i in range(384)])
    return embeddings


@pytest.fixture()
def mock_chromadb() -> MagicMock:
    """Inject a mock chromadb module so ChromaVectorStore can be imported."""
    mock = MagicMock()
    original = sys.modules.get("chromadb")
    sys.modules["chromadb"] = mock
    yield mock
    # Restore
    if original is not None:
        sys.modules["chromadb"] = original
    else:
        sys.modules.pop("chromadb", None)
    # Clear cached chroma_store module so next test gets a fresh import
    sys.modules.pop("src.vectorstore.chroma_store", None)


def _make_store(
    mock_chromadb: MagicMock,
    mock_collection: MagicMock,
    tmp_path: Path,
    collection_name: str = "test",
) -> object:
    """Helper to build a ChromaVectorStore with mocked chromadb."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    # Import after mocking
    from src.vectorstore.chroma_store import ChromaVectorStore

    return ChromaVectorStore(
        collection_name=collection_name,
        persist_dir=str(tmp_path / "chroma"),
    )


class TestIngestionPipeline:
    """End-to-end test of the full ingestion pipeline with mocked embeddings."""

    def test_full_pipeline(self, mock_chromadb: MagicMock, tmp_path: Path) -> None:
        """Load sample docs → chunk → preprocess → embed → store → search."""

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        store = _make_store(mock_chromadb, mock_collection, tmp_path, "test_full")

        # --- Step 1: Load documents ---
        loader = MarkdownLoader()
        all_docs: list[Document] = []
        for md_file in sorted(SAMPLE_DIR.glob("*.md")):
            docs = loader.load(str(md_file))
            all_docs.extend(docs)

        assert len(all_docs) == 4

        # --- Step 2: Preprocess ---
        pipeline = PreprocessingPipeline()
        processed = pipeline.run(all_docs)
        assert len(processed) == 4
        for doc in processed:
            assert doc.metadata.get("fingerprint")
            assert doc.content.strip()

        # --- Step 3: Chunk ---
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(processed)

        assert len(chunks) > len(processed)
        for chunk in chunks:
            assert chunk.metadata.get("chunk_index") is not None
            assert chunk.metadata.get("parent_doc_id")
            assert chunk.metadata.get("source")

        # --- Step 4: Embed (mocked) ---
        embeddings = _fake_embeddings([c.content for c in chunks])
        assert len(embeddings) == len(chunks)
        assert len(embeddings[0]) == 384

        # --- Step 5: Store ---
        ids = store.add_documents(chunks, embeddings)
        assert len(ids) == len(chunks)
        assert mock_collection.upsert.called

    def test_search_returns_results(self, mock_chromadb: MagicMock, tmp_path: Path) -> None:
        """Test that search returns ranked results with metadata."""

        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "documents": [["chunk about RAG", "chunk about agents", "chunk about LLMs"]],
            "metadatas": [[
                {"source": "rag_systems.md", "file_type": "markdown", "chunk_index": "0"},
                {"source": "ai_agents.md", "file_type": "markdown", "chunk_index": "2"},
                {"source": "rag_systems.md", "file_type": "markdown", "chunk_index": "5"},
            ]],
            "distances": [[0.1, 0.3, 0.5]],
        }

        store = _make_store(mock_chromadb, mock_collection, tmp_path, "test_search")

        query_embedding = [0.5] * 384
        results = store.search(query_embedding, k=3)

        assert len(results) == 3
        assert results[0].score > results[1].score > results[2].score
        assert results[0].rank == 0
        assert results[0].document.content == "chunk about RAG"
        assert results[0].document.metadata["source"] == "rag_systems.md"

    def test_metadata_filtering(self, mock_chromadb: MagicMock, tmp_path: Path) -> None:
        """Test that metadata filters are passed to ChromaDB."""

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["filtered result"]],
            "metadatas": [[{"source": "rag_systems.md", "file_type": "markdown"}]],
            "distances": [[0.2]],
        }

        store = _make_store(mock_chromadb, mock_collection, tmp_path, "test_filter")

        query_embedding = [0.5] * 384
        metadata_filter = {"source": "rag_systems.md"}
        results = store.search(query_embedding, k=5, where=metadata_filter)

        assert len(results) == 1
        # Verify the filter was passed to ChromaDB
        call_kwargs = mock_collection.query.call_args
        assert call_kwargs.kwargs.get("where") == metadata_filter

    def test_pipeline_chunk_metadata_integrity(self) -> None:
        """Verify metadata flows correctly through load → preprocess → chunk."""
        loader = MarkdownLoader()
        docs = loader.load(str(SAMPLE_DIR / "rag_systems.md"))
        assert docs[0].metadata["file_type"] == "markdown"

        pipeline = PreprocessingPipeline()
        processed = pipeline.run(docs)
        assert processed[0].metadata["file_type"] == "markdown"
        assert processed[0].metadata.get("fingerprint")

        chunker = RecursiveChunker(chunk_size=256, chunk_overlap=30)
        chunks = chunker.chunk(processed)

        for chunk in chunks:
            assert chunk.metadata["file_type"] == "markdown"
            assert "rag_systems.md" in chunk.metadata["source"]
            assert chunk.metadata.get("fingerprint")
            assert "chunk_index" in chunk.metadata
            assert "parent_doc_id" in chunk.metadata
