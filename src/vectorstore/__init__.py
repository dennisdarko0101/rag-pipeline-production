from src.vectorstore.base import SearchResult, VectorStore

__all__ = [
    "VectorStore",
    "SearchResult",
    "ChromaVectorStore",
]


def __getattr__(name: str) -> object:
    """Lazy import ChromaVectorStore to avoid importing chromadb at module load time."""
    if name == "ChromaVectorStore":
        from src.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
