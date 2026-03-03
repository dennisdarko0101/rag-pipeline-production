"""Text preprocessing utilities for the ingestion pipeline."""

import hashlib
import re
import unicodedata

from src.models.document import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Common header/footer patterns to strip
_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^page\s+\d+\s*(of\s+\d+)?$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^©.*$", re.MULTILINE),
    re.compile(r"^all rights reserved.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*confidential\s*$", re.IGNORECASE | re.MULTILINE),
]

_DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4})\b"
    ),
]


def clean_text(text: str) -> str:
    """Remove extra whitespace, normalize unicode, and strip headers/footers.

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text.
    """
    # Normalize unicode characters (e.g. curly quotes → straight quotes)
    text = unicodedata.normalize("NFKC", text)

    # Strip header/footer patterns
    for pattern in _HEADER_FOOTER_PATTERNS:
        text = pattern.sub("", text)

    # Collapse multiple blank lines into two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces (but not newlines) into single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def extract_metadata(text: str) -> dict[str, str | list[str]]:
    """Pull titles, headers, and dates from text.

    Args:
        text: Document text to extract metadata from.

    Returns:
        Dictionary with extracted metadata fields.
    """
    metadata: dict[str, str | list[str]] = {}

    lines = text.splitlines()

    # Extract markdown-style title (first H1)
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            metadata["title"] = stripped.removeprefix("# ").strip()
            break

    # Extract all headers
    headers: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            # Count heading level
            level = 0
            for ch in stripped:
                if ch == "#":
                    level += 1
                else:
                    break
            header_text = stripped[level:].strip()
            if header_text:
                headers.append(header_text)
    if headers:
        metadata["headers"] = headers

    # Extract dates
    dates: list[str] = []
    for pattern in _DATE_PATTERNS:
        dates.extend(pattern.findall(text))
    if dates:
        metadata["dates"] = list(dict.fromkeys(dates))  # deduplicate, preserve order

    return metadata


def generate_fingerprint(text: str) -> str:
    """Generate a content fingerprint for deduplication.

    Uses SHA-256 on normalized text (lowercased, whitespace-collapsed).

    Args:
        text: Document text to fingerprint.

    Returns:
        Hex digest of the content hash.
    """
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate(documents: list[Document]) -> list[Document]:
    """Remove duplicate documents based on content fingerprints.

    Args:
        documents: List of documents, each should have a 'fingerprint' in metadata.

    Returns:
        Deduplicated list of documents.
    """
    seen: set[str] = set()
    unique: list[Document] = []

    for doc in documents:
        fp = doc.metadata.get("fingerprint") or generate_fingerprint(doc.content)
        if fp not in seen:
            seen.add(fp)
            unique.append(doc)
        else:
            logger.debug("duplicate_removed", source=doc.source, fingerprint=fp[:12])

    removed = len(documents) - len(unique)
    if removed:
        logger.info("deduplication_complete", removed=removed, remaining=len(unique))

    return unique


class PreprocessingPipeline:
    """Chains preprocessing steps: clean -> extract_metadata -> deduplicate."""

    def run(self, documents: list[Document]) -> list[Document]:
        """Run the full preprocessing pipeline on a list of documents.

        Args:
            documents: Raw documents from loaders.

        Returns:
            Cleaned, enriched, and deduplicated documents.
        """
        logger.info("preprocessing_start", doc_count=len(documents))
        processed: list[Document] = []

        for doc in documents:
            # 1. Clean text
            cleaned = clean_text(doc.content)
            if not cleaned:
                logger.debug("empty_after_cleaning", source=doc.source)
                continue

            # 2. Extract metadata
            extracted = extract_metadata(cleaned)

            # 3. Generate fingerprint
            fingerprint = generate_fingerprint(cleaned)

            # Build updated document
            updated_metadata = {**doc.metadata, **extracted, "fingerprint": fingerprint}
            processed.append(
                Document(
                    doc_id=doc.doc_id,
                    content=cleaned,
                    metadata=updated_metadata,
                    created_at=doc.created_at,
                )
            )

        # 4. Deduplicate
        result = deduplicate(processed)

        logger.info(
            "preprocessing_complete",
            input_docs=len(documents),
            output_docs=len(result),
        )
        return result
