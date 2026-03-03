"""Tests for embedding models."""

from unittest.mock import MagicMock, patch

import openai
import pytest

from src.embeddings.embedder import BaseEmbedder, OpenAIEmbedder


def _make_embedding_response(embeddings: list[list[float]]) -> MagicMock:
    """Create a mock OpenAI embeddings response."""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=emb, index=i) for i, emb in enumerate(embeddings)
    ]
    return mock_response


FAKE_EMBEDDING = [0.1] * 1536


class TestOpenAIEmbedder:
    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_embed_text(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response([FAKE_EMBEDDING])
        mock_cls.return_value = mock_client

        embedder = OpenAIEmbedder(api_key="test-key")
        result = embedder.embed_text("Hello world")

        assert len(result) == 1536
        assert result == FAKE_EMBEDDING
        mock_client.embeddings.create.assert_called_once()

    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_embed_batch(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        texts = ["text one", "text two", "text three"]
        embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response(embeddings)
        mock_cls.return_value = mock_client

        embedder = OpenAIEmbedder(api_key="test-key", batch_size=100)
        results = embedder.embed_batch(texts)

        assert len(results) == 3
        assert results[0] == [0.1] * 1536
        assert results[2] == [0.3] * 1536
        # Single batch since batch_size=100 > 3 texts
        mock_client.embeddings.create.assert_called_once()

    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_batch_splits_correctly(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        """Test that large batches are split into sub-batches."""
        mock_client = MagicMock()
        # Return 2 embeddings per call (batch_size=2)
        mock_client.embeddings.create.side_effect = [
            _make_embedding_response([[0.1] * 1536, [0.2] * 1536]),
            _make_embedding_response([[0.3] * 1536]),
        ]
        mock_cls.return_value = mock_client

        embedder = OpenAIEmbedder(api_key="test-key", batch_size=2)
        results = embedder.embed_batch(["a", "b", "c"])

        assert len(results) == 3
        assert mock_client.embeddings.create.call_count == 2

    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_embed_empty_batch(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        embedder = OpenAIEmbedder(api_key="test-key")
        results = embedder.embed_batch([])
        assert results == []

    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_retry_on_api_error(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        """Test that retry logic triggers on transient API errors."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = [
            openai.APIConnectionError(request=MagicMock()),
            _make_embedding_response([FAKE_EMBEDDING]),
        ]
        mock_cls.return_value = mock_client

        embedder = OpenAIEmbedder(api_key="test-key")
        result = embedder.embed_text("test retry")

        assert len(result) == 1536
        assert mock_client.embeddings.create.call_count == 2

    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_retry_exhausted_raises(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        """Test that after 3 retries the error propagates."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        mock_cls.return_value = mock_client

        embedder = OpenAIEmbedder(api_key="test-key")
        with pytest.raises(openai.APIConnectionError):
            embedder.embed_text("fail forever")

        assert mock_client.embeddings.create.call_count == 3

    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_embedding_dimensions(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        small_emb = [0.5] * 512
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = _make_embedding_response([small_emb])
        mock_cls.return_value = mock_client

        embedder = OpenAIEmbedder(api_key="test-key", dimensions=512)
        result = embedder.embed_text("test dimensions")

        assert len(result) == 512
        # Verify dimensions was passed to the API
        call_kwargs = mock_client.embeddings.create.call_args
        assert call_kwargs.kwargs.get("dimensions") == 512 or call_kwargs[1].get("dimensions") == 512

    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_token_counting(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        embedder = OpenAIEmbedder(api_key="test-key")
        count = embedder.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    @patch("src.embeddings.embedder.openai.OpenAI")
    @patch("src.embeddings.embedder.openai.AsyncOpenAI")
    def test_truncation(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        embedder = OpenAIEmbedder(api_key="test-key")
        # A very long text should be truncated
        long_text = "word " * 100000
        truncated = embedder.truncate_text(long_text)
        assert embedder.count_tokens(truncated) <= 8191

    def test_base_class_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbedder()  # type: ignore[abstract]
