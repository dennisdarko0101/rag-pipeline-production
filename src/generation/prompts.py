"""Prompt templates for RAG pipeline: system prompts, user templates, and formatters."""

from src.vectorstore.base import SearchResult

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based ONLY on the provided context. "
    "Follow these rules strictly:\n"
    "1. Answer ONLY using information from the provided context.\n"
    "2. Cite your sources using [Source: filename, chunk N] format after each claim.\n"
    "3. If the context does not contain enough information to answer the question, "
    'say "I don\'t have enough information to answer this question."\n'
    "4. Be concise and factual. Do not speculate or add information beyond the context.\n"
    "5. If multiple sources support a claim, cite all of them."
)

RAG_USER_PROMPT_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer the question based only on the context above. "
    "Cite sources using [Source: filename, chunk N] format."
)

QUERY_EXPANSION_PROMPT = (
    "Generate {num_variants} alternative phrasings of this search query. "
    "Each should capture the same intent but use different words.\n\n"
    "Query: {query}\n\n"
    "Return ONLY the alternative queries, one per line, without numbering or bullets."
)

HYDE_PROMPT = (
    "Write a short, detailed passage that would directly answer this question. "
    "Write it as if it were a paragraph from a technical document.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)

# Maximum context length in characters to prevent exceeding LLM token limits
MAX_CONTEXT_CHARS = 12000


def format_context(results: list[SearchResult], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Format search results into a context string for the LLM.

    Each result is formatted as a numbered section with source info.
    Truncates if total context exceeds max_chars.

    Args:
        results: Search results to format as context.
        max_chars: Maximum total character length.

    Returns:
        Formatted context string.
    """
    sections: list[str] = []
    total_len = 0

    for i, result in enumerate(results):
        source = result.document.metadata.get("source", "unknown")
        chunk_idx = result.document.metadata.get("chunk_index", i)

        section = f"[{i + 1}] Source: {source}, Chunk {chunk_idx}\n{result.document.content}"

        if total_len + len(section) > max_chars:
            # Truncate this section to fit within the limit
            remaining = max_chars - total_len
            if remaining > 50:
                section = section[:remaining] + "\n[...truncated]"
                sections.append(section)
            break

        sections.append(section)
        total_len += len(section) + 2  # +2 for separator newlines

    return "\n\n".join(sections)


def format_rag_prompt(question: str, results: list[SearchResult]) -> str:
    """Build the full RAG user prompt from question and results.

    Args:
        question: The user's question.
        results: Retrieved search results.

    Returns:
        Formatted user prompt string.
    """
    context = format_context(results)
    return RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)


def format_query_expansion_prompt(query: str, num_variants: int = 3) -> str:
    """Build the query expansion prompt.

    Args:
        query: The original search query.
        num_variants: Number of alternative queries to generate.

    Returns:
        Formatted prompt string.
    """
    return QUERY_EXPANSION_PROMPT.format(query=query, num_variants=num_variants)


def format_hyde_prompt(query: str) -> str:
    """Build the HyDE prompt.

    Args:
        query: The search query.

    Returns:
        Formatted prompt string.
    """
    return HYDE_PROMPT.format(query=query)
