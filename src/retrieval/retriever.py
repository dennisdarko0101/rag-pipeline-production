"""Retrieval strategies: semantic, BM25, and hybrid."""

from abc import ABC, abstractmethod
from time import perf_counter

from rank_bm25 import BM25Okapi

from src.embeddings.embedder import BaseEmbedder
from src.models.document import Document
from src.utils.logger import get_logger
from src.vectorstore.base import SearchResult, VectorStore

logger = get_logger(__name__)


class BaseRetriever(ABC):
    """Base class for all retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[SearchResult]:
        """Retrieve documents relevant to the query.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of SearchResult objects ranked by relevance.
        """


class SemanticRetriever(BaseRetriever):
    """Dense vector retrieval using an embedder and vector store."""

    def __init__(self, embedder: BaseEmbedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._store = vector_store
        logger.info("semantic_retriever_init")

    def retrieve(
        self,
        query: str,
        k: int = 10,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Embed the query and search the vector store.

        Args:
            query: The search query.
            k: Number of results to return.
            where: Optional metadata filter.

        Returns:
            Ranked search results.
        """
        start = perf_counter()
        query_embedding = self._embedder.embed_text(query)
        results = self._store.search(query_embedding, k=k, where=where)
        elapsed = perf_counter() - start

        logger.info(
            "semantic_retrieve",
            query=query[:80],
            k=k,
            num_results=len(results),
            latency_ms=round(elapsed * 1000, 1),
        )
        return results


class BM25Retriever(BaseRetriever):
    """Sparse keyword retrieval using BM25 (Okapi BM25)."""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        if documents:
            self.index(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def index(self, documents: list[Document]) -> None:
        """Build the BM25 index from documents.

        Args:
            documents: List of documents to index.
        """
        self._documents = list(documents)
        tokenized = [self._tokenize(doc.content) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("bm25_index_built", num_documents=len(self._documents))

    def retrieve(self, query: str, k: int = 10) -> list[SearchResult]:
        """Search using BM25 scoring.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            Ranked search results with BM25 scores normalized to 0-1.
        """
        if self._bm25 is None or not self._documents:
            logger.warning("bm25_empty_index")
            return []

        start = perf_counter()
        tokens = self._tokenize(query)
        raw_scores = self._bm25.get_scores(tokens)

        # Normalize scores to 0-1 range
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        normalized = [s / max_score for s in raw_scores]

        # Sort by score descending and take top-k
        scored_pairs = sorted(
            enumerate(normalized),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = [
            SearchResult(
                document=self._documents[idx],
                score=score,
                rank=rank,
            )
            for rank, (idx, score) in enumerate(scored_pairs)
            if score > 0
        ]

        elapsed = perf_counter() - start
        logger.info(
            "bm25_retrieve",
            query=query[:80],
            k=k,
            num_results=len(results),
            latency_ms=round(elapsed * 1000, 1),
        )
        return results


class HybridRetriever(BaseRetriever):
    """Combines semantic and BM25 retrieval using Reciprocal Rank Fusion.

    RRF formula: score = sum(weight / (rrf_k + rank)) for each retriever.
    """

    def __init__(
        self,
        semantic: SemanticRetriever,
        bm25: BM25Retriever,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> None:
        self._semantic = semantic
        self._bm25 = bm25
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        self._rrf_k = rrf_k
        logger.info(
            "hybrid_retriever_init",
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            rrf_k=rrf_k,
        )

    def retrieve(self, query: str, k: int = 10) -> list[SearchResult]:
        """Retrieve using both semantic and BM25, fused with RRF.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            Ranked search results with fused scores.
        """
        start = perf_counter()

        # Fetch more candidates from each retriever to have enough for fusion
        fetch_k = k * 3
        semantic_results = self._semantic.retrieve(query, k=fetch_k)
        bm25_results = self._bm25.retrieve(query, k=fetch_k)

        # Calculate RRF scores keyed by doc_id
        fused_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for result in semantic_results:
            doc_id = result.document.doc_id
            doc_map[doc_id] = result.document
            rrf_score = self._semantic_weight / (self._rrf_k + result.rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score

        for result in bm25_results:
            doc_id = result.document.doc_id
            doc_map[doc_id] = result.document
            rrf_score = self._keyword_weight / (self._rrf_k + result.rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score

        # Sort by fused score descending and take top-k
        sorted_ids = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)[:k]

        results = [
            SearchResult(
                document=doc_map[doc_id],
                score=fused_scores[doc_id],
                rank=rank,
            )
            for rank, doc_id in enumerate(sorted_ids)
        ]

        elapsed = perf_counter() - start
        logger.info(
            "hybrid_retrieve",
            query=query[:80],
            k=k,
            semantic_candidates=len(semantic_results),
            bm25_candidates=len(bm25_results),
            num_results=len(results),
            latency_ms=round(elapsed * 1000, 1),
        )
        return results
