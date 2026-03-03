"""Tests for prompt templates and formatting."""

from src.generation.prompts import (
    HYDE_PROMPT,
    MAX_CONTEXT_CHARS,
    QUERY_EXPANSION_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
    format_context,
    format_hyde_prompt,
    format_query_expansion_prompt,
    format_rag_prompt,
)
from src.models.document import Document
from src.vectorstore.base import SearchResult


def _make_result(
    doc_id: str,
    content: str,
    source: str = "test.md",
    chunk_index: int = 0,
    score: float = 0.9,
    rank: int = 0,
) -> SearchResult:
    return SearchResult(
        document=Document(
            doc_id=doc_id,
            content=content,
            metadata={"source": source, "chunk_index": chunk_index},
        ),
        score=score,
        rank=rank,
    )


class TestPromptConstants:
    def test_system_prompt_has_citation_format(self) -> None:
        assert "[Source:" in RAG_SYSTEM_PROMPT

    def test_system_prompt_mentions_context_only(self) -> None:
        assert "ONLY" in RAG_SYSTEM_PROMPT

    def test_system_prompt_has_insufficient_info(self) -> None:
        assert "don't have enough information" in RAG_SYSTEM_PROMPT

    def test_user_template_has_placeholders(self) -> None:
        assert "{context}" in RAG_USER_PROMPT_TEMPLATE
        assert "{question}" in RAG_USER_PROMPT_TEMPLATE

    def test_query_expansion_has_placeholders(self) -> None:
        assert "{query}" in QUERY_EXPANSION_PROMPT
        assert "{num_variants}" in QUERY_EXPANSION_PROMPT

    def test_hyde_has_placeholder(self) -> None:
        assert "{query}" in HYDE_PROMPT


class TestFormatContext:
    def test_formats_single_result(self) -> None:
        results = [_make_result("d1", "RAG systems combine retrieval and generation.")]
        context = format_context(results)
        assert "[1] Source: test.md, Chunk 0" in context
        assert "RAG systems combine" in context

    def test_formats_multiple_results(self) -> None:
        results = [
            _make_result("d1", "First chunk", source="a.md", chunk_index=0, rank=0),
            _make_result("d2", "Second chunk", source="b.md", chunk_index=3, rank=1),
        ]
        context = format_context(results)
        assert "[1] Source: a.md, Chunk 0" in context
        assert "[2] Source: b.md, Chunk 3" in context

    def test_truncates_long_context(self) -> None:
        # Create results that exceed MAX_CONTEXT_CHARS
        results = [
            _make_result(f"d{i}", "x" * 5000, chunk_index=i, rank=i)
            for i in range(10)
        ]
        context = format_context(results, max_chars=500)
        assert len(context) <= 600  # Allow some overhead for headers and truncation marker

    def test_truncation_adds_marker(self) -> None:
        results = [
            _make_result("d1", "x" * 5000, rank=0),
            _make_result("d2", "y" * 5000, rank=1),
        ]
        context = format_context(results, max_chars=500)
        assert "[...truncated]" in context

    def test_empty_results(self) -> None:
        context = format_context([])
        assert context == ""

    def test_uses_default_max_chars(self) -> None:
        # Just verify it doesn't crash with default
        results = [_make_result("d1", "content")]
        context = format_context(results)
        assert len(context) <= MAX_CONTEXT_CHARS + 100


class TestFormatRagPrompt:
    def test_includes_question_and_context(self) -> None:
        results = [_make_result("d1", "RAG is cool.")]
        prompt = format_rag_prompt("What is RAG?", results)
        assert "What is RAG?" in prompt
        assert "RAG is cool." in prompt

    def test_includes_citation_instruction(self) -> None:
        results = [_make_result("d1", "content")]
        prompt = format_rag_prompt("question", results)
        assert "Cite sources" in prompt

    def test_empty_results_produces_empty_context(self) -> None:
        prompt = format_rag_prompt("question", [])
        assert "Context:\n\n" in prompt


class TestFormatQueryExpansionPrompt:
    def test_includes_query_and_count(self) -> None:
        prompt = format_query_expansion_prompt("What is RAG?", num_variants=5)
        assert "What is RAG?" in prompt
        assert "5" in prompt

    def test_default_num_variants(self) -> None:
        prompt = format_query_expansion_prompt("test")
        assert "3" in prompt


class TestFormatHydePrompt:
    def test_includes_query(self) -> None:
        prompt = format_hyde_prompt("How do transformers work?")
        assert "How do transformers work?" in prompt

    def test_includes_passage_instruction(self) -> None:
        prompt = format_hyde_prompt("test")
        assert "Passage:" in prompt
