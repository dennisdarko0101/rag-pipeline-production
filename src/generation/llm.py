"""LLM abstraction layer: Claude, OpenAI, fallback, and factory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter

import anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Tracks cumulative token usage across LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from a single call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_calls += 1

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
        }


class BaseLLM(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The generated text response.
        """

    @abstractmethod
    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        """Async version of generate.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.

        Returns:
            The generated text response.
        """

    @property
    @abstractmethod
    def usage(self) -> TokenUsage:
        """Return cumulative token usage."""


class ClaudeLLM(BaseLLM):
    """Anthropic Claude LLM with retry logic and token tracking."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model or settings.llm_model
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens or settings.llm_max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or settings.anthropic_api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=api_key or settings.anthropic_api_key)
        self._usage = TokenUsage()

        logger.info(
            "claude_llm_init",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    @property
    def usage(self) -> TokenUsage:
        return self._usage

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_api(self, prompt: str, system: str | None = None) -> anthropic.types.Message:
        """Call the Anthropic API with retry logic."""
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        return self._client.messages.create(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _acall_api(self, prompt: str, system: str | None = None) -> anthropic.types.Message:
        """Async call to the Anthropic API with retry logic."""
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        return await self._async_client.messages.create(**kwargs)

    def generate(self, prompt: str, system: str | None = None) -> str:
        start = perf_counter()
        response = self._call_api(prompt, system)
        text = response.content[0].text
        elapsed = perf_counter() - start

        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens
        self._usage.record(input_tok, output_tok)

        logger.info(
            "claude_generate",
            model=self._model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=round(elapsed * 1000, 1),
        )
        return text

    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        start = perf_counter()
        response = await self._acall_api(prompt, system)
        text = response.content[0].text
        elapsed = perf_counter() - start

        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens
        self._usage.record(input_tok, output_tok)

        logger.info(
            "claude_agenerate",
            model=self._model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=round(elapsed * 1000, 1),
        )
        return text


class OpenAILLM(BaseLLM):
    """OpenAI GPT LLM with retry logic and token tracking."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model or settings.llm_fallback_model
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens or settings.llm_max_tokens
        self._client = openai.OpenAI(api_key=api_key or settings.openai_api_key)
        self._async_client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self._usage = TokenUsage()

        logger.info(
            "openai_llm_init",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    @property
    def usage(self) -> TokenUsage:
        return self._usage

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_api(
        self,
        prompt: str,
        system: str | None = None,
    ) -> openai.types.chat.ChatCompletion:
        """Call the OpenAI API with retry logic."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=messages,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _acall_api(
        self,
        prompt: str,
        system: str | None = None,
    ) -> openai.types.chat.ChatCompletion:
        """Async call to the OpenAI API with retry logic."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self._async_client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=messages,
        )

    def generate(self, prompt: str, system: str | None = None) -> str:
        start = perf_counter()
        response = self._call_api(prompt, system)
        text = response.choices[0].message.content or ""
        elapsed = perf_counter() - start

        usage = response.usage
        input_tok = usage.prompt_tokens if usage else 0
        output_tok = usage.completion_tokens if usage else 0
        self._usage.record(input_tok, output_tok)

        logger.info(
            "openai_generate",
            model=self._model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=round(elapsed * 1000, 1),
        )
        return text

    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        start = perf_counter()
        response = await self._acall_api(prompt, system)
        text = response.choices[0].message.content or ""
        elapsed = perf_counter() - start

        usage = response.usage
        input_tok = usage.prompt_tokens if usage else 0
        output_tok = usage.completion_tokens if usage else 0
        self._usage.record(input_tok, output_tok)

        logger.info(
            "openai_agenerate",
            model=self._model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=round(elapsed * 1000, 1),
        )
        return text


@dataclass
class _FallbackStats:
    """Internal tracking for FallbackLLM."""

    primary_successes: int = 0
    fallback_successes: int = 0
    total_failures: int = 0


class FallbackLLM(BaseLLM):
    """Tries the primary LLM, falls back to secondary on failure.

    Tracks which provider was used and fallback frequency.
    """

    def __init__(self, primary: BaseLLM, secondary: BaseLLM) -> None:
        self._primary = primary
        self._secondary = secondary
        self._stats = _FallbackStats()
        self._combined_usage = TokenUsage()
        logger.info("fallback_llm_init")

    @property
    def usage(self) -> TokenUsage:
        return self._combined_usage

    @property
    def fallback_stats(self) -> dict:
        return {
            "primary_successes": self._stats.primary_successes,
            "fallback_successes": self._stats.fallback_successes,
            "total_failures": self._stats.total_failures,
            "fallback_rate": (
                self._stats.fallback_successes
                / max(self._stats.primary_successes + self._stats.fallback_successes, 1)
            ),
        }

    def generate(self, prompt: str, system: str | None = None) -> str:
        # Try primary
        try:
            result = self._primary.generate(prompt, system)
            self._stats.primary_successes += 1
            self._combined_usage.record(
                self._primary.usage.input_tokens - (self._combined_usage.input_tokens - self._combined_usage.input_tokens),
                0,
            )
            # Sync usage from primary
            self._combined_usage = TokenUsage(
                input_tokens=self._primary.usage.input_tokens + self._secondary.usage.input_tokens,
                output_tokens=self._primary.usage.output_tokens + self._secondary.usage.output_tokens,
                total_calls=self._primary.usage.total_calls + self._secondary.usage.total_calls,
            )
            logger.info("fallback_llm_generate", provider="primary")
            return result
        except Exception as e:
            logger.warning("fallback_llm_primary_failed", error=str(e))

        # Try secondary
        try:
            result = self._secondary.generate(prompt, system)
            self._stats.fallback_successes += 1
            self._combined_usage = TokenUsage(
                input_tokens=self._primary.usage.input_tokens + self._secondary.usage.input_tokens,
                output_tokens=self._primary.usage.output_tokens + self._secondary.usage.output_tokens,
                total_calls=self._primary.usage.total_calls + self._secondary.usage.total_calls,
            )
            logger.info("fallback_llm_generate", provider="secondary")
            return result
        except Exception as e:
            self._stats.total_failures += 1
            logger.error("fallback_llm_all_failed", error=str(e))
            raise

    async def agenerate(self, prompt: str, system: str | None = None) -> str:
        try:
            result = await self._primary.agenerate(prompt, system)
            self._stats.primary_successes += 1
            self._combined_usage = TokenUsage(
                input_tokens=self._primary.usage.input_tokens + self._secondary.usage.input_tokens,
                output_tokens=self._primary.usage.output_tokens + self._secondary.usage.output_tokens,
                total_calls=self._primary.usage.total_calls + self._secondary.usage.total_calls,
            )
            logger.info("fallback_llm_agenerate", provider="primary")
            return result
        except Exception as e:
            logger.warning("fallback_llm_primary_failed", error=str(e))

        try:
            result = await self._secondary.agenerate(prompt, system)
            self._stats.fallback_successes += 1
            self._combined_usage = TokenUsage(
                input_tokens=self._primary.usage.input_tokens + self._secondary.usage.input_tokens,
                output_tokens=self._primary.usage.output_tokens + self._secondary.usage.output_tokens,
                total_calls=self._primary.usage.total_calls + self._secondary.usage.total_calls,
            )
            logger.info("fallback_llm_agenerate", provider="secondary")
            return result
        except Exception as e:
            self._stats.total_failures += 1
            logger.error("fallback_llm_all_failed", error=str(e))
            raise


class LLMFactory:
    """Creates LLM instances from configuration."""

    @staticmethod
    def create(
        provider: str = "claude",
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> BaseLLM:
        """Create an LLM instance.

        Args:
            provider: LLM provider ("claude", "openai", "fallback").
            model: Model name override.
            api_key: API key override.
            temperature: Temperature override.
            max_tokens: Max tokens override.

        Returns:
            A configured BaseLLM instance.
        """
        if provider == "claude":
            return ClaudeLLM(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "openai":
            return OpenAILLM(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "fallback":
            primary = ClaudeLLM(temperature=temperature, max_tokens=max_tokens)
            secondary = OpenAILLM(temperature=temperature, max_tokens=max_tokens)
            return FallbackLLM(primary=primary, secondary=secondary)

        raise ValueError(f"Unknown LLM provider: {provider}. Options: claude, openai, fallback")
