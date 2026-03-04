"""Tests for embedding cache."""

from pathlib import Path
from unittest.mock import MagicMock

from src.embeddings.cache import CachedEmbedder, EmbeddingCache

FAKE_EMBEDDING = [0.1, 0.2, 0.3] * 100  # 300-dim fake embedding


class TestEmbeddingCache:
    def test_cache_miss(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path)
        result = cache.get("hello", "text-embedding-3-small")
        assert result is None

    def test_cache_hit(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path)
        cache.set("hello", "text-embedding-3-small", FAKE_EMBEDDING)
        result = cache.get("hello", "text-embedding-3-small")
        assert result == FAKE_EMBEDDING

    def test_different_text_different_key(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path)
        cache.set("hello", "model-a", [1.0, 2.0])
        assert cache.get("world", "model-a") is None

    def test_different_model_different_key(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path)
        cache.set("hello", "model-a", [1.0, 2.0])
        assert cache.get("hello", "model-b") is None

    def test_cache_key_deterministic(self) -> None:
        key1 = EmbeddingCache._cache_key("hello", "model")
        key2 = EmbeddingCache._cache_key("hello", "model")
        assert key1 == key2

    def test_cache_key_unique(self) -> None:
        key1 = EmbeddingCache._cache_key("hello", "model")
        key2 = EmbeddingCache._cache_key("world", "model")
        assert key1 != key2

    def test_stats(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path)

        # One miss
        cache.get("text1", "model")
        # One set + hit
        cache.set("text2", "model", [1.0])
        cache.get("text2", "model")

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_cached"] == 1
        assert stats["hit_rate"] == 50.0

    def test_stats_zero_calls(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path)
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_ttl_expired(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path, ttl_seconds=0)
        cache.set("hello", "model", [1.0, 2.0])
        # TTL=0 means already expired
        result = cache.get("hello", "model")
        assert result is None

    def test_ttl_not_expired(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path, ttl_seconds=3600)
        cache.set("hello", "model", [1.0, 2.0])
        result = cache.get("hello", "model")
        assert result == [1.0, 2.0]

    def test_clear(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path)
        cache.set("text1", "model", [1.0])
        cache.set("text2", "model", [2.0])

        deleted = cache.clear()
        assert deleted == 2
        assert cache.get("text1", "model") is None
        assert cache.stats()["total_cached"] == 0

    def test_overwrite_existing(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_dir=tmp_path)
        cache.set("hello", "model", [1.0])
        cache.set("hello", "model", [2.0])
        result = cache.get("hello", "model")
        assert result == [2.0]


class TestCachedEmbedder:
    def _make_mock_embedder(self) -> MagicMock:
        """Create a mock BaseEmbedder."""
        mock = MagicMock()
        mock.model = "test-model"
        mock.embed_text.return_value = FAKE_EMBEDDING
        mock.embed_batch.return_value = [FAKE_EMBEDDING, FAKE_EMBEDDING]
        return mock

    def test_cache_miss_forwards_to_embedder(self, tmp_path: Path) -> None:
        mock_embedder = self._make_mock_embedder()
        cache = EmbeddingCache(cache_dir=tmp_path)
        cached = CachedEmbedder(embedder=mock_embedder, cache=cache)

        result = cached.embed_text("hello")
        assert result == FAKE_EMBEDDING
        mock_embedder.embed_text.assert_called_once_with("hello")

    def test_cache_hit_skips_embedder(self, tmp_path: Path) -> None:
        mock_embedder = self._make_mock_embedder()
        cache = EmbeddingCache(cache_dir=tmp_path)
        cached = CachedEmbedder(embedder=mock_embedder, cache=cache)

        # First call: cache miss → forwards to embedder
        cached.embed_text("hello")
        # Second call: cache hit → skips embedder
        result = cached.embed_text("hello")

        assert result == FAKE_EMBEDDING
        assert mock_embedder.embed_text.call_count == 1

    def test_batch_uses_cache_for_hits(self, tmp_path: Path) -> None:
        mock_embedder = self._make_mock_embedder()
        mock_embedder.embed_batch.return_value = [FAKE_EMBEDDING]
        cache = EmbeddingCache(cache_dir=tmp_path)
        cached = CachedEmbedder(embedder=mock_embedder, cache=cache)

        # Pre-cache one text
        cache.set("cached_text", "test-model", FAKE_EMBEDDING)

        # Batch with one cached and one uncached
        results = cached.embed_batch(["cached_text", "new_text"])

        assert len(results) == 2
        # Only the uncached text should be sent to the embedder
        mock_embedder.embed_batch.assert_called_once_with(["new_text"])

    def test_cache_stats_accessible(self, tmp_path: Path) -> None:
        mock_embedder = self._make_mock_embedder()
        cache = EmbeddingCache(cache_dir=tmp_path)
        cached = CachedEmbedder(embedder=mock_embedder, cache=cache)

        cached.embed_text("hello")
        cached.embed_text("hello")

        stats = cached.cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_empty_batch(self, tmp_path: Path) -> None:
        mock_embedder = self._make_mock_embedder()
        cache = EmbeddingCache(cache_dir=tmp_path)
        cached = CachedEmbedder(embedder=mock_embedder, cache=cache)

        results = cached.embed_batch([])
        assert results == []
        mock_embedder.embed_batch.assert_not_called()
