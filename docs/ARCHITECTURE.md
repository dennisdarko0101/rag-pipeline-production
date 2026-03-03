# Architecture

## System Overview

The RAG Pipeline processes user queries through a multi-stage pipeline:

1. **Ingestion** - Documents are loaded, chunked, embedded, and stored in ChromaDB
2. **Retrieval** - Hybrid search (semantic + BM25) retrieves candidates, then cross-encoder reranks
3. **Generation** - LLM generates answers grounded in retrieved context with citations
4. **Evaluation** - RAGAS metrics measure quality across faithfulness, relevancy, and recall

## Component Details

*To be expanded as components are implemented.*
