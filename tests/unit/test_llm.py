"""Tests for LLM abstraction layer."""

from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import openai
import pytest

from src.generation.llm import (
    BaseLLM,
    ClaudeLLM,
    FallbackLLM,
    LLMFactory,
    OpenAILLM,
    TokenUsage,
)


def _mock_claude_response(text: str = "Hello!", input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    """Create a mock Anthropic Message response."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


def _mock_openai_response(text: str = "Hello!", prompt_tokens: int = 10, completion_tokens: int = 5) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=text))]
    response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response


class TestTokenUsage:
    def test_initial_state(self) -> None:
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.total_calls == 0

    def test_record_accumulates(self) -> None:
        usage = TokenUsage()
        usage.record(100, 50)
        usage.record(200, 75)
        assert usage.input_tokens == 300
        assert usage.output_tokens == 125
        assert usage.total_tokens == 425
        assert usage.total_calls == 2

    def test_to_dict(self) -> None:
        usage = TokenUsage()
        usage.record(10, 5)
        d = usage.to_dict()
        assert d["input_tokens"] == 10
        assert d["output_tokens"] == 5
        assert d["total_tokens"] == 15
        assert d["total_calls"] == 1


class TestClaudeLLM:
    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    @patch("src.generation.llm.anthropic.Anthropic")
    def test_generate(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_claude_response("Test answer", 15, 8)
        mock_cls.return_value = mock_client

        llm = ClaudeLLM(api_key="test-key")
        result = llm.generate("What is RAG?")

        assert result == "Test answer"
        mock_client.messages.create.assert_called_once()

    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    @patch("src.generation.llm.anthropic.Anthropic")
    def test_generate_with_system(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_claude_response()
        mock_cls.return_value = mock_client

        llm = ClaudeLLM(api_key="test-key")
        llm.generate("Hello", system="You are helpful")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"

    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    @patch("src.generation.llm.anthropic.Anthropic")
    def test_token_tracking(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_claude_response("Answer", 20, 10)
        mock_cls.return_value = mock_client

        llm = ClaudeLLM(api_key="test-key")
        llm.generate("Q1")
        llm.generate("Q2")

        assert llm.usage.input_tokens == 40
        assert llm.usage.output_tokens == 20
        assert llm.usage.total_calls == 2

    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    @patch("src.generation.llm.anthropic.Anthropic")
    def test_retry_on_api_error(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            anthropic.APIConnectionError(request=MagicMock()),
            _mock_claude_response("Retry success"),
        ]
        mock_cls.return_value = mock_client

        llm = ClaudeLLM(api_key="test-key")
        result = llm.generate("Test retry")

        assert result == "Retry success"
        assert mock_client.messages.create.call_count == 2

    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    @patch("src.generation.llm.anthropic.Anthropic")
    def test_retry_exhausted_raises(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(
            request=MagicMock()
        )
        mock_cls.return_value = mock_client

        llm = ClaudeLLM(api_key="test-key")
        with pytest.raises(anthropic.APIConnectionError):
            llm.generate("Fail forever")

        assert mock_client.messages.create.call_count == 3

    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    @patch("src.generation.llm.anthropic.Anthropic")
    def test_custom_params(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_claude_response()
        mock_cls.return_value = mock_client

        llm = ClaudeLLM(api_key="test-key", model="claude-3-haiku-20240307", temperature=0.5, max_tokens=500)
        llm.generate("Test")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-haiku-20240307"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    @patch("src.generation.llm.anthropic.Anthropic")
    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    async def test_agenerate(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        mock_async_client = AsyncMock()
        mock_async_client.messages.create.return_value = _mock_claude_response("Async answer", 12, 6)
        mock_async_cls.return_value = mock_async_client

        llm = ClaudeLLM(api_key="test-key")
        result = await llm.agenerate("Async question")

        assert result == "Async answer"
        assert llm.usage.total_calls == 1


class TestOpenAILLM:
    @patch("src.generation.llm.openai.AsyncOpenAI")
    @patch("src.generation.llm.openai.OpenAI")
    def test_generate(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("GPT answer", 12, 7)
        mock_cls.return_value = mock_client

        llm = OpenAILLM(api_key="test-key")
        result = llm.generate("What is RAG?")

        assert result == "GPT answer"
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.generation.llm.openai.AsyncOpenAI")
    @patch("src.generation.llm.openai.OpenAI")
    def test_generate_with_system(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response()
        mock_cls.return_value = mock_client

        llm = OpenAILLM(api_key="test-key")
        llm.generate("Hello", system="You are helpful")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    @patch("src.generation.llm.openai.AsyncOpenAI")
    @patch("src.generation.llm.openai.OpenAI")
    def test_token_tracking(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("Answer", 25, 15)
        mock_cls.return_value = mock_client

        llm = OpenAILLM(api_key="test-key")
        llm.generate("Q1")
        llm.generate("Q2")

        assert llm.usage.input_tokens == 50
        assert llm.usage.output_tokens == 30
        assert llm.usage.total_calls == 2

    @patch("src.generation.llm.openai.AsyncOpenAI")
    @patch("src.generation.llm.openai.OpenAI")
    def test_retry_on_api_error(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            openai.APIConnectionError(request=MagicMock()),
            _mock_openai_response("Retry success"),
        ]
        mock_cls.return_value = mock_client

        llm = OpenAILLM(api_key="test-key")
        result = llm.generate("Test retry")

        assert result == "Retry success"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("src.generation.llm.openai.AsyncOpenAI")
    @patch("src.generation.llm.openai.OpenAI")
    def test_retry_exhausted_raises(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        mock_cls.return_value = mock_client

        llm = OpenAILLM(api_key="test-key")
        with pytest.raises(openai.APIConnectionError):
            llm.generate("Fail forever")

        assert mock_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    @patch("src.generation.llm.openai.OpenAI")
    @patch("src.generation.llm.openai.AsyncOpenAI")
    async def test_agenerate(self, mock_async_cls: MagicMock, mock_cls: MagicMock) -> None:
        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create.return_value = _mock_openai_response("Async GPT", 10, 5)
        mock_async_cls.return_value = mock_async_client

        llm = OpenAILLM(api_key="test-key")
        result = await llm.agenerate("Async question")

        assert result == "Async GPT"
        assert llm.usage.total_calls == 1


class TestFallbackLLM:
    def _make_mock_llm(self, response: str = "Answer") -> MagicMock:
        mock = MagicMock(spec=BaseLLM)
        mock.generate.return_value = response
        mock.usage = TokenUsage()
        mock.usage.record(10, 5)
        return mock

    def test_uses_primary_on_success(self) -> None:
        primary = self._make_mock_llm("Primary answer")
        secondary = self._make_mock_llm("Secondary answer")

        fallback = FallbackLLM(primary=primary, secondary=secondary)
        result = fallback.generate("Test")

        assert result == "Primary answer"
        primary.generate.assert_called_once()
        secondary.generate.assert_not_called()

    def test_falls_back_on_primary_failure(self) -> None:
        primary = self._make_mock_llm()
        primary.generate.side_effect = RuntimeError("Primary down")
        secondary = self._make_mock_llm("Fallback answer")

        fallback = FallbackLLM(primary=primary, secondary=secondary)
        result = fallback.generate("Test")

        assert result == "Fallback answer"
        secondary.generate.assert_called_once()

    def test_raises_when_both_fail(self) -> None:
        primary = self._make_mock_llm()
        primary.generate.side_effect = RuntimeError("Primary down")
        secondary = self._make_mock_llm()
        secondary.generate.side_effect = RuntimeError("Secondary down")

        fallback = FallbackLLM(primary=primary, secondary=secondary)
        with pytest.raises(RuntimeError, match="Secondary down"):
            fallback.generate("Test")

    def test_fallback_stats_tracking(self) -> None:
        primary = self._make_mock_llm("OK")
        secondary = self._make_mock_llm("Fallback")

        fallback = FallbackLLM(primary=primary, secondary=secondary)

        # Successful primary call
        fallback.generate("Q1")
        assert fallback.fallback_stats["primary_successes"] == 1
        assert fallback.fallback_stats["fallback_successes"] == 0

        # Failed primary → fallback
        primary.generate.side_effect = RuntimeError("Oops")
        fallback.generate("Q2")
        assert fallback.fallback_stats["primary_successes"] == 1
        assert fallback.fallback_stats["fallback_successes"] == 1

    def test_fallback_rate(self) -> None:
        primary = self._make_mock_llm("OK")
        secondary = self._make_mock_llm("Fallback")

        fallback = FallbackLLM(primary=primary, secondary=secondary)

        # 1 primary, 0 fallback → rate 0.0
        fallback.generate("Q1")
        assert fallback.fallback_stats["fallback_rate"] == 0.0

        # 1 primary, 1 fallback → rate 0.5
        primary.generate.side_effect = RuntimeError("Oops")
        fallback.generate("Q2")
        assert fallback.fallback_stats["fallback_rate"] == 0.5

    def test_total_failures_tracked(self) -> None:
        primary = self._make_mock_llm()
        primary.generate.side_effect = RuntimeError("Down")
        secondary = self._make_mock_llm()
        secondary.generate.side_effect = RuntimeError("Also down")

        fallback = FallbackLLM(primary=primary, secondary=secondary)

        with pytest.raises(RuntimeError):
            fallback.generate("Q1")

        assert fallback.fallback_stats["total_failures"] == 1


class TestLLMFactory:
    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    @patch("src.generation.llm.anthropic.Anthropic")
    def test_create_claude(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        llm = LLMFactory.create("claude", api_key="test")
        assert isinstance(llm, ClaudeLLM)

    @patch("src.generation.llm.openai.AsyncOpenAI")
    @patch("src.generation.llm.openai.OpenAI")
    def test_create_openai(self, mock_cls: MagicMock, mock_async_cls: MagicMock) -> None:
        llm = LLMFactory.create("openai", api_key="test")
        assert isinstance(llm, OpenAILLM)

    @patch("src.generation.llm.openai.AsyncOpenAI")
    @patch("src.generation.llm.openai.OpenAI")
    @patch("src.generation.llm.anthropic.AsyncAnthropic")
    @patch("src.generation.llm.anthropic.Anthropic")
    def test_create_fallback(self, *mocks: MagicMock) -> None:
        llm = LLMFactory.create("fallback")
        assert isinstance(llm, FallbackLLM)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create("unknown")

    def test_base_class_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]
