"""LLM generation, RAG chain, prompt templates, and response parsing."""

from src.generation.chain import RAGChain, RAGResponse, Source
from src.generation.llm import (
    BaseLLM,
    ClaudeLLM,
    FallbackLLM,
    LLMFactory,
    OpenAILLM,
    TokenUsage,
)
from src.generation.prompts import (
    HYDE_PROMPT,
    QUERY_EXPANSION_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    format_context,
    format_hyde_prompt,
    format_query_expansion_prompt,
    format_rag_prompt,
)
from src.generation.response_parser import Citation, parse_citations, process_response

__all__ = [
    "BaseLLM",
    "Citation",
    "ClaudeLLM",
    "FallbackLLM",
    "HYDE_PROMPT",
    "LLMFactory",
    "OpenAILLM",
    "QUERY_EXPANSION_PROMPT",
    "RAG_SYSTEM_PROMPT",
    "RAG_USER_PROMPT_TEMPLATE",
    "RAGChain",
    "RAGResponse",
    "Source",
    "TokenUsage",
    "format_context",
    "format_hyde_prompt",
    "format_query_expansion_prompt",
    "format_rag_prompt",
    "parse_citations",
    "process_response",
]
