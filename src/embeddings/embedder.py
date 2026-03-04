"""Embedding models for converting text to dense vector representations."""

from abc import ABC, abstractmethod
from time import perf_counter

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Token limits per model
_MODEL_TOKEN_LIMITS: dict[str, int] = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}


class BaseEmbedder(ABC):
    """Base class for all embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """

    @abstractmethod
    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed_batch.

        Args:
            texts: List of texts to embed.

        Returns:
            A list of embedding vectors.
        """


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding model wrapper with batching, retry, and token counting."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 100,
        dimensions: int | None = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self.batch_size = batch_size
        self.dimensions = dimensions or settings.embedding_dimension
        self._client = openai.OpenAI(api_key=api_key or settings.openai_api_key)
        self._async_client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self._token_limit = _MODEL_TOKEN_LIMITS.get(self.model, 8191)

        try:
            self._encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(
            "openai_embedder_init",
            model=self.model,
            batch_size=self.batch_size,
            dimensions=self.dimensions,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        return len(self._encoding.encode(text))

    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within the model's token limit.

        Args:
            text: The text to truncate.

        Returns:
            Truncated text that fits within the token limit.
        """
        tokens = self._encoding.encode(text)
        if len(tokens) <= self._token_limit:
            return text
        logger.warning(
            "text_truncated",
            original_tokens=len(tokens),
            limit=self._token_limit,
        )
        return self._encoding.decode(tokens[: self._token_limit])

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API with retry logic.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors.
        """
        response = self._client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _acall_api(self, texts: list[str]) -> list[list[float]]:
        """Async call to the OpenAI embeddings API with retry logic.

        Args:
            texts: Batch of texts to embed.

        Returns:
            List of embedding vectors.
        """
        response = await self._async_client.embeddings.create(
            input=texts,
            model=self.model,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector.
        """
        start = perf_counter()
        truncated = self.truncate_text(text)
        result = self._call_api([truncated])[0]
        elapsed = perf_counter() - start

        logger.info(
            "embed_text",
            model=self.model,
            text_len=len(text),
            tokens=self.count_tokens(truncated),
            dimensions=len(result),
            latency_ms=round(elapsed * 1000, 1),
        )
        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, splitting into sub-batches as needed.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input.
        """
        if not texts:
            return []

        start = perf_counter()
        truncated = [self.truncate_text(t) for t in texts]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(truncated), self.batch_size):
            batch = truncated[i : i + self.batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        elapsed = perf_counter() - start
        logger.info(
            "embed_batch",
            model=self.model,
            total_texts=len(texts),
            batches=(len(texts) + self.batch_size - 1) // self.batch_size,
            latency_ms=round(elapsed * 1000, 1),
        )
        return all_embeddings

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Async embed a batch of texts, splitting into sub-batches as needed.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input.
        """
        if not texts:
            return []

        start = perf_counter()
        truncated = [self.truncate_text(t) for t in texts]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(truncated), self.batch_size):
            batch = truncated[i : i + self.batch_size]
            embeddings = await self._acall_api(batch)
            all_embeddings.extend(embeddings)

        elapsed = perf_counter() - start
        logger.info(
            "aembed_batch",
            model=self.model,
            total_texts=len(texts),
            batches=(len(texts) + self.batch_size - 1) // self.batch_size,
            latency_ms=round(elapsed * 1000, 1),
        )
        return all_embeddings
