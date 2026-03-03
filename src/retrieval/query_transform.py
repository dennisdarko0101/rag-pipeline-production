"""Query transformation strategies: expansion, HyDE, and multi-query."""

from time import perf_counter

import anthropic
import openai

from src.config.settings import settings
from src.embeddings.embedder import BaseEmbedder
from src.retrieval.retriever import BaseRetriever
from src.utils.logger import get_logger
from src.vectorstore.base import SearchResult

logger = get_logger(__name__)


def _call_llm(prompt: str) -> str:
    """Call the configured LLM (Claude primary, GPT-4o fallback).

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        The LLM response text.
    """
    # Try Claude first
    if settings.anthropic_api_key:
        try:
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning("claude_call_failed", error=str(e))

    # Fallback to OpenAI
    if settings.openai_api_key:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.llm_fallback_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    raise RuntimeError("No LLM API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")


class QueryExpander:
    """Uses an LLM to generate alternative phrasings of a query."""

    def __init__(self, num_variants: int = 3) -> None:
        self._num_variants = num_variants
        logger.info("query_expander_init", num_variants=num_variants)

    def expand(self, query: str) -> list[str]:
        """Generate alternative phrasings of the query.

        Args:
            query: The original search query.

        Returns:
            List of alternative query strings (includes the original).
        """
        start = perf_counter()

        prompt = (
            f"Generate {self._num_variants} alternative phrasings of this search query. "
            f"Each should capture the same intent but use different words.\n\n"
            f"Query: {query}\n\n"
            f"Return ONLY the alternative queries, one per line, without numbering or bullets."
        )

        response = _call_llm(prompt)
        variants = [
            line.strip()
            for line in response.strip().splitlines()
            if line.strip()
        ][:self._num_variants]

        # Always include the original query
        result = [query] + variants

        elapsed = perf_counter() - start
        logger.info(
            "query_expanded",
            original=query[:80],
            num_variants=len(variants),
            latency_ms=round(elapsed * 1000, 1),
        )
        return result


class HyDE:
    """Hypothetical Document Embedding.

    Asks the LLM to generate a hypothetical answer, then embeds that
    answer instead of the raw query for semantic search. The hypothesis
    is closer in embedding space to real relevant documents than the
    short query.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder
        logger.info("hyde_init")

    def generate_embedding(self, query: str) -> list[float]:
        """Generate a hypothetical document and embed it.

        Args:
            query: The search query.

        Returns:
            Embedding vector of the hypothetical answer.
        """
        start = perf_counter()

        prompt = (
            f"Write a short, detailed passage that would directly answer this question. "
            f"Write it as if it were a paragraph from a technical document.\n\n"
            f"Question: {query}\n\n"
            f"Passage:"
        )

        hypothesis = _call_llm(prompt)
        embedding = self._embedder.embed_text(hypothesis)

        elapsed = perf_counter() - start
        logger.info(
            "hyde_generated",
            query=query[:80],
            hypothesis_len=len(hypothesis),
            latency_ms=round(elapsed * 1000, 1),
        )
        return embedding


class MultiQueryRetriever:
    """Expands a query into multiple variants, retrieves for each, and merges.

    Deduplicates results by doc_id, keeping the highest score for each.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        expander: QueryExpander,
    ) -> None:
        self._retriever = retriever
        self._expander = expander
        logger.info("multi_query_retriever_init")

    def retrieve(self, query: str, k: int = 10) -> list[SearchResult]:
        """Expand the query and retrieve for each variant.

        Args:
            query: The original search query.
            k: Number of final results to return.

        Returns:
            Deduplicated, merged results sorted by best score.
        """
        start = perf_counter()

        queries = self._expander.expand(query)

        # Collect all results, keeping best score per doc_id
        best_scores: dict[str, float] = {}
        doc_map: dict[str, SearchResult] = {}

        for q in queries:
            results = self._retriever.retrieve(q, k=k)
            for result in results:
                doc_id = result.document.doc_id
                if doc_id not in best_scores or result.score > best_scores[doc_id]:
                    best_scores[doc_id] = result.score
                    doc_map[doc_id] = result

        # Sort by score descending and take top-k
        sorted_ids = sorted(best_scores, key=lambda d: best_scores[d], reverse=True)[:k]
        merged = [
            SearchResult(
                document=doc_map[doc_id].document,
                score=best_scores[doc_id],
                rank=rank,
            )
            for rank, doc_id in enumerate(sorted_ids)
        ]

        elapsed = perf_counter() - start
        logger.info(
            "multi_query_retrieve",
            original_query=query[:80],
            num_queries=len(queries),
            num_results=len(merged),
            latency_ms=round(elapsed * 1000, 1),
        )
        return merged
