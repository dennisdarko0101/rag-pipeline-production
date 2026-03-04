"""Parse and validate citations from LLM responses."""

import re
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Matches [Source: filename, chunk N] or [Source: filename, Chunk N]
_CITATION_PATTERN = re.compile(r"\[Source:\s*([^,\]]+?),\s*[Cc]hunk\s*(\d+)\]")


@dataclass
class Citation:
    """A parsed citation from an LLM response."""

    source: str
    chunk_index: int
    raw_text: str


def parse_citations(text: str) -> list[Citation]:
    """Extract all citations from an LLM response.

    Args:
        text: The LLM response text.

    Returns:
        List of parsed Citation objects.
    """
    citations: list[Citation] = []
    for match in _CITATION_PATTERN.finditer(text):
        citations.append(
            Citation(
                source=match.group(1).strip(),
                chunk_index=int(match.group(2)),
                raw_text=match.group(0),
            )
        )
    return citations


def validate_citations(
    citations: list[Citation],
    valid_sources: set[str],
) -> tuple[list[Citation], list[Citation]]:
    """Validate citations against actually retrieved sources.

    Args:
        citations: Parsed citations from the response.
        valid_sources: Set of source filenames that were actually retrieved.

    Returns:
        Tuple of (valid_citations, invalid_citations).
    """
    valid: list[Citation] = []
    invalid: list[Citation] = []

    for citation in citations:
        if citation.source in valid_sources:
            valid.append(citation)
        else:
            invalid.append(citation)

    if invalid:
        logger.warning(
            "invalid_citations_found",
            count=len(invalid),
            invalid_sources=[c.source for c in invalid],
        )

    return valid, invalid


def strip_invalid_citations(text: str, invalid_citations: list[Citation]) -> str:
    """Remove invalid citations from the response text.

    Args:
        text: The original LLM response.
        invalid_citations: Citations that failed validation.

    Returns:
        Cleaned text with invalid citations removed.
    """
    for citation in invalid_citations:
        text = text.replace(citation.raw_text, "")

    # Clean up double spaces left by removal
    text = re.sub(r"  +", " ", text)
    # Clean up empty lines left by removal
    text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
    return text.strip()


def process_response(text: str, valid_sources: set[str]) -> tuple[str, list[Citation]]:
    """Full pipeline: parse, validate, and clean citations.

    Args:
        text: The raw LLM response.
        valid_sources: Set of valid source filenames.

    Returns:
        Tuple of (cleaned text, list of valid citations).
    """
    citations = parse_citations(text)
    valid, invalid = validate_citations(citations, valid_sources)

    cleaned = strip_invalid_citations(text, invalid) if invalid else text

    logger.info(
        "response_processed",
        total_citations=len(citations),
        valid_citations=len(valid),
        invalid_citations=len(invalid),
    )
    return cleaned, valid
