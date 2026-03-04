"""File-based embedding cache for avoiding redundant API calls."""

import hashlib
import json
import time
from pathlib import Path

from src.embeddings.embedder import BaseEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CACHE_DIR = Path("./data/embedding_cache")


class EmbeddingCache:
    """File-based cache for embedding vectors.

    Each embedding is stored as a JSON file keyed by the SHA-256 hash of
    (model_name + text). Supports optional TTL-based expiration.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0
        logger.info("embedding_cache_init", cache_dir=str(self._cache_dir), ttl=self._ttl)

    @staticmethod
    def _cache_key(text: str, model: str) -> str:
        """Generate a cache key from text and model name.

        Args:
            text: The input text.
            model: The embedding model name.

        Returns:
            SHA-256 hex digest.
        """
        raw = f"{model}:{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        """Get the file path for a cache key, using 2-char prefix subdirectory."""
        subdir = self._cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.json"

    def get(self, text: str, model: str) -> list[float] | None:
        """Retrieve a cached embedding.

        Args:
            text: The original text.
            model: The embedding model name.

        Returns:
            The cached embedding vector, or None on cache miss.
        """
        key = self._cache_key(text, model)
        path = self._path_for_key(key)

        if not path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._misses += 1
            return None

        # Check TTL
        if self._ttl is not None:
            created = data.get("created_at", 0)
            if time.time() - created > self._ttl:
                path.unlink(missing_ok=True)
                self._misses += 1
                return None

        self._hits += 1
        return data["embedding"]

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        """Store an embedding in the cache.

        Args:
            text: The original text.
            model: The embedding model name.
            embedding: The embedding vector to cache.
        """
        key = self._cache_key(text, model)
        path = self._path_for_key(key)

        data = {
            "model": model,
            "embedding": embedding,
            "created_at": time.time(),
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def stats(self) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Dictionary with hits, misses, and total cached files.
        """
        total_files = sum(1 for _ in self._cache_dir.rglob("*.json"))
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_cached": total_files,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1) * 100, 1),
        }

    def clear(self) -> int:
        """Delete all cached embeddings.

        Returns:
            Number of files deleted.
        """
        count = 0
        for f in self._cache_dir.rglob("*.json"):
            f.unlink()
            count += 1
        logger.info("cache_cleared", files_deleted=count)
        return count


class CachedEmbedder(BaseEmbedder):
    """Wraps any BaseEmbedder and adds file-based caching.

    Cache hits skip the API call entirely. Misses are forwarded to the
    wrapped embedder and the results are stored for future use.
    """

    def __init__(self, embedder: BaseEmbedder, cache: EmbeddingCache) -> None:
        self._embedder = embedder
        self._cache = cache
        self._model = getattr(embedder, "model", "unknown")
        logger.info("cached_embedder_init", model=self._model)

    @property
    def cache(self) -> EmbeddingCache:
        """Access the underlying cache for stats."""
        return self._cache

    def embed_text(self, text: str) -> list[float]:
        """Embed text, checking cache first.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector.
        """
        cached = self._cache.get(text, self._model)
        if cached is not None:
            return cached

        embedding = self._embedder.embed_text(text)
        self._cache.set(text, self._model, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, using cache where possible.

        Texts with cache hits are returned directly. Only cache misses
        are sent to the underlying embedder.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._cache.get(text, self._model)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self._embedder.embed_batch(uncached_texts)
            for idx, embedding in zip(uncached_indices, new_embeddings, strict=True):
                results[idx] = embedding
                self._cache.set(texts[idx], self._model, embedding)

        logger.info(
            "cached_embed_batch",
            total=len(texts),
            cache_hits=len(texts) - len(uncached_texts),
            api_calls=len(uncached_texts),
        )
        return results  # type: ignore[return-value]

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Async embed with caching.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors in the same order as input.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text, self._model)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = await self._embedder.aembed_batch(uncached_texts)
            for idx, embedding in zip(uncached_indices, new_embeddings, strict=True):
                results[idx] = embedding
                self._cache.set(texts[idx], self._model, embedding)

        return results  # type: ignore[return-value]
