from prometheus_client import Counter, Histogram, Info

app_info = Info("rag_pipeline", "RAG Pipeline application info")
app_info.info({"version": "0.1.0"})

query_counter = Counter(
    "rag_queries_total",
    "Total number of RAG queries",
    ["status"],
)

query_latency = Histogram(
    "rag_query_duration_seconds",
    "RAG query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ingestion_counter = Counter(
    "rag_documents_ingested_total",
    "Total number of documents ingested",
    ["doc_type"],
)

retrieval_latency = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

embedding_latency = Histogram(
    "rag_embedding_duration_seconds",
    "Embedding computation latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0],
)
