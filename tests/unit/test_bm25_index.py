"""Tests for the shared BM25 index and its wiring into the query path.

These guard against a regression where BM25 is constructed empty and never
indexed, which silently degrades "hybrid" retrieval to semantic-only. The
mocked query-route tests miss this because they patch out the whole chain, so
the checks here exercise the real wiring.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models.document import Document
from src.retrieval import bm25_index
from src.retrieval.retriever import BM25Retriever, HybridRetriever

SAMPLE_DOCS = [
    Document(
        doc_id="d1",
        content="Transformers use self-attention over token sequences.",
        metadata={"source": "transformers.md"},
    ),
    Document(
        doc_id="d2",
        content="BM25 is a sparse keyword ranking function for retrieval.",
        metadata={"source": "retrieval.md"},
    ),
    Document(
        doc_id="d3",
        content="Retrieval augmented generation grounds answers in documents.",
        metadata={"source": "rag.md"},
    ),
]


@pytest.fixture(autouse=True)
def _reset_shared_index() -> None:
    """Each test starts and ends with a clean shared index."""
    bm25_index.reset()
    yield
    bm25_index.reset()


def _fake_store(docs: list[Document] | None = None) -> MagicMock:
    store = MagicMock()
    store.get_all_documents.return_value = list(docs if docs is not None else SAMPLE_DOCS)
    return store


class TestSharedBM25Index:
    def test_builds_from_store_and_retrieves(self) -> None:
        bm25 = bm25_index.get_bm25_retriever(_fake_store())
        assert bm25.num_documents == len(SAMPLE_DOCS)
        results = bm25.retrieve("keyword ranking", k=3)
        assert results, "BM25 must return candidates once indexed from the corpus"
        assert results[0].document.doc_id == "d2"

    def test_caches_single_warm_instance(self) -> None:
        store = _fake_store()
        first = bm25_index.get_bm25_retriever(store)
        second = bm25_index.get_bm25_retriever(store)
        assert first is second
        store.get_all_documents.assert_called_once()

    def test_add_documents_extends_index(self) -> None:
        bm25 = bm25_index.get_bm25_retriever(_fake_store())
        before = bm25.num_documents
        bm25_index.add_documents(
            [
                Document(
                    doc_id="d4",
                    content="Cross encoder reranking improves precision.",
                    metadata={"source": "rerank.md"},
                )
            ]
        )
        assert bm25.num_documents == before + 1
        results = bm25.retrieve("reranking precision", k=3)
        assert any(r.document.doc_id == "d4" for r in results)

    def test_add_before_build_is_noop_then_lazy_build_works(self) -> None:
        # Index not built yet: add is a no-op (docs are already in the store).
        bm25_index.add_documents(SAMPLE_DOCS)
        bm25 = bm25_index.get_bm25_retriever(_fake_store())
        assert bm25.num_documents == len(SAMPLE_DOCS)

    def test_reset_forces_rebuild(self) -> None:
        bm25_index.get_bm25_retriever(_fake_store())
        bm25_index.reset()
        store2 = _fake_store()
        bm25_index.get_bm25_retriever(store2)
        store2.get_all_documents.assert_called_once()


class TestQueryPathEngagesHybrid:
    """Regression guard: the real /api/v1/query wiring must index BM25."""

    def test_build_rag_chain_indexes_bm25_from_store(self) -> None:
        from src.api.routes import query as query_route

        fake_store = _fake_store()
        with (
            patch("src.vectorstore.chroma_store.ChromaVectorStore", return_value=fake_store),
            patch("src.embeddings.embedder.OpenAIEmbedder", return_value=MagicMock()),
            patch("src.generation.llm.LLMFactory.create", return_value=MagicMock()),
        ):
            chain = query_route._build_rag_chain("fallback", rerank=False)

        hybrid = chain._retriever
        assert isinstance(hybrid, HybridRetriever)
        bm25 = hybrid._bm25
        assert isinstance(bm25, BM25Retriever)

        # The core regression: BM25 must be populated at query time, otherwise
        # hybrid retrieval has silently degraded to semantic-only.
        assert bm25.num_documents == len(SAMPLE_DOCS), (
            "BM25 index is empty at query time; hybrid retrieval degraded to semantic-only"
        )
        assert bm25.retrieve("self-attention", k=3), (
            "BM25 returned no candidates despite an indexed corpus"
        )

    def test_hybrid_fuses_both_retrievers_with_rrf(self) -> None:
        """Both retrievers contribute and RRF merges their rankings."""
        from src.retrieval.retriever import SemanticRetriever
        from src.vectorstore.base import SearchResult

        bm25 = bm25_index.get_bm25_retriever(_fake_store())

        # Semantic returns a doc that BM25 ranks lower for this query, so a
        # fused result that contains both proves RRF combined the two lists.
        semantic = MagicMock(spec=SemanticRetriever)
        semantic.retrieve.return_value = [
            SearchResult(document=SAMPLE_DOCS[2], score=0.99, rank=0),
        ]

        hybrid = HybridRetriever(semantic=semantic, bm25=bm25)
        fused = hybrid.retrieve("keyword ranking function", k=5)

        semantic.retrieve.assert_called_once()
        fused_ids = {r.document.doc_id for r in fused}
        assert "d3" in fused_ids, "semantic-only result missing from fused output"
        assert "d2" in fused_ids, "BM25-only result missing from fused output"
