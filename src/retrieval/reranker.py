"""Reranking strategies: cross-encoder and LLM-based."""

from abc import ABC, abstractmethod
from time import perf_counter

from src.retrieval.query_transform import _call_llm
from src.utils.logger import get_logger
from src.vectorstore.base import SearchResult

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Base class for all reranking strategies."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results by relevance to the query.

        Args:
            query: The search query.
            results: Initial search results to rerank.
            top_k: Number of top results to return.

        Returns:
            Reranked list of SearchResult objects.
        """


class CrossEncoderReranker(BaseReranker):
    """Reranks using a cross-encoder model (sentence-transformers).

    The cross-encoder scores each (query, document) pair directly,
    which is more accurate than bi-encoder similarity but slower.
    Uses lazy model loading — only loads when first called.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None
        logger.info("cross_encoder_reranker_init", model=model_name)

    def _load_model(self) -> None:
        """Lazy-load the cross-encoder model on first use."""
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_name)
        logger.info("cross_encoder_model_loaded", model=self._model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Score each (query, document) pair with the cross-encoder.

        Args:
            query: The search query.
            results: Initial search results to rerank.
            top_k: Number of top results to return.

        Returns:
            Top-k results sorted by cross-encoder score.
        """
        if not results:
            return []

        start = perf_counter()
        self._load_model()

        # Build (query, document) pairs for scoring
        pairs = [[query, r.document.content] for r in results]
        scores = self._model.predict(pairs)

        # Pair results with new scores, sort descending, take top_k
        scored = sorted(
            zip(results, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        reranked = [
            SearchResult(
                document=result.document,
                score=float(score),
                rank=rank,
            )
            for rank, (result, score) in enumerate(scored)
        ]

        elapsed = perf_counter() - start
        logger.info(
            "cross_encoder_rerank",
            query=query[:80],
            num_input=len(results),
            num_results=len(reranked),
            latency_ms=round(elapsed * 1000, 1),
        )
        return reranked


class LLMReranker(BaseReranker):
    """Reranks using an LLM to score relevance.

    More expensive than cross-encoder but can leverage the LLM's
    deeper understanding of relevance. Scores each document 1-10.
    """

    def __init__(self, batch_size: int = 5) -> None:
        self._batch_size = batch_size
        logger.info("llm_reranker_init", batch_size=batch_size)

    def _score_batch(self, query: str, results: list[SearchResult]) -> list[float]:
        """Score a batch of documents against the query using the LLM.

        Args:
            query: The search query.
            results: Batch of results to score.

        Returns:
            List of relevance scores (1-10) for each result.
        """
        docs_text = "\n\n".join(
            f"[Document {i + 1}]\n{r.document.content[:500]}"
            for i, r in enumerate(results)
        )

        prompt = (
            f"Rate the relevance of each document to the query on a scale of 1-10.\n"
            f"1 = completely irrelevant, 10 = perfectly relevant.\n\n"
            f"Query: {query}\n\n"
            f"{docs_text}\n\n"
            f"Return ONLY the scores as comma-separated numbers in order. "
            f"Example: 8,3,7,5,9"
        )

        response = _call_llm(prompt)

        # Parse scores from LLM response
        scores: list[float] = []
        for token in response.strip().split(","):
            token = token.strip()
            try:
                score = float(token)
                scores.append(min(max(score, 1.0), 10.0))  # Clamp to 1-10
            except ValueError:
                scores.append(5.0)  # Default score if parsing fails

        # Pad or truncate to match input length
        while len(scores) < len(results):
            scores.append(5.0)
        scores = scores[: len(results)]

        return scores

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Score each document with the LLM and return top-k.

        Args:
            query: The search query.
            results: Initial search results to rerank.
            top_k: Number of top results to return.

        Returns:
            Top-k results sorted by LLM relevance score.
        """
        if not results:
            return []

        start = perf_counter()

        # Score in batches to minimize API calls
        all_scores: list[float] = []
        for i in range(0, len(results), self._batch_size):
            batch = results[i : i + self._batch_size]
            batch_scores = self._score_batch(query, batch)
            all_scores.extend(batch_scores)

        # Pair results with scores, sort descending, take top_k
        scored = sorted(
            zip(results, all_scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        reranked = [
            SearchResult(
                document=result.document,
                score=float(score),
                rank=rank,
            )
            for rank, (result, score) in enumerate(scored)
        ]

        elapsed = perf_counter() - start
        logger.info(
            "llm_rerank",
            query=query[:80],
            num_input=len(results),
            num_results=len(reranked),
            latency_ms=round(elapsed * 1000, 1),
        )
        return reranked
