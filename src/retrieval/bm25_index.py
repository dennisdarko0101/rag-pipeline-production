"""Process-wide BM25 keyword index shared across requests.

Dense retrieval keeps its corpus in ChromaDB, but BM25 needs the documents in
memory to score keyword matches. Rebuilding that from the vector store on every
request would be wasteful, so the index is built once (lazily, from the
persisted documents) and kept warm for the life of the process. Ingestion
appends new documents to it so sparse and dense retrieval stay in sync.

Both the ``/api/v1/query`` and ``/api/v1/ingest`` routes go through this module,
which is what makes hybrid retrieval actually engage at runtime rather than
silently degrading to semantic-only.
"""

from threading import Lock

from src.models.document import Document
from src.retrieval.retriever import BM25Retriever
from src.utils.logger import get_logger
from src.vectorstore.base import VectorStore

logger = get_logger(__name__)

_retriever: BM25Retriever | None = None
_lock = Lock()


def get_bm25_retriever(store: VectorStore) -> BM25Retriever:
    """Return the shared BM25 retriever, building it from the store if needed.

    The first call loads every persisted document into the index. Subsequent
    calls return the same warm instance.

    Args:
        store: Vector store to load the corpus from on first build.

    Returns:
        A BM25Retriever indexed over the stored corpus.
    """
    global _retriever
    with _lock:
        if _retriever is None:
            documents = store.get_all_documents()
            _retriever = BM25Retriever(documents=documents)
            logger.info("bm25_shared_index_built", num_documents=len(documents))
        return _retriever


def add_documents(documents: list[Document]) -> None:
    """Add newly ingested documents to the shared index.

    If the index has not been built yet, this is a no-op: the documents are
    already in the vector store, so the next ``get_bm25_retriever`` call will
    pick them up when it builds lazily.

    Args:
        documents: Newly ingested document chunks.
    """
    global _retriever
    if not documents:
        return
    with _lock:
        if _retriever is None:
            return
        _retriever.add(documents)
        logger.info("bm25_shared_index_updated", added=len(documents))


def reset() -> None:
    """Drop the cached index so the next access rebuilds it.

    Used by tests and after a full re-seed.
    """
    global _retriever
    with _lock:
        _retriever = None
