#!/usr/bin/env bash
set -euo pipefail

echo "=== RAG Pipeline: Seeding Vector Database ==="
echo ""

python3 -c "
import sys
from pathlib import Path

from src.config.settings import settings
from src.ingestion.loader import MarkdownLoader
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.preprocessor import PreprocessingPipeline
from src.embeddings.embedder import OpenAIEmbedder

# Check API key
if not settings.openai_api_key:
    print('ERROR: OPENAI_API_KEY is not set. Add it to .env file.')
    sys.exit(1)

sample_dir = Path('data/sample_docs')
if not sample_dir.exists():
    print(f'ERROR: Sample docs directory not found: {sample_dir}')
    sys.exit(1)

# Step 1: Load documents
print('Step 1: Loading documents...')
loader = MarkdownLoader()
all_docs = []
for md_file in sorted(sample_dir.glob('*.md')):
    docs = loader.load(str(md_file))
    all_docs.extend(docs)
    print(f'  Loaded: {md_file.name} ({len(docs[0].content)} chars)')
print(f'  Total documents: {len(all_docs)}')

# Step 2: Preprocess
print('Step 2: Preprocessing...')
pipeline = PreprocessingPipeline()
processed = pipeline.run(all_docs)
print(f'  Preprocessed: {len(processed)} documents')

# Step 3: Chunk
print('Step 3: Chunking...')
chunker = RecursiveChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
chunks = chunker.chunk(processed)
print(f'  Created {len(chunks)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})')

# Step 4: Embed
print('Step 4: Embedding with OpenAI...')
embedder = OpenAIEmbedder()
texts = [c.content for c in chunks]
embeddings = embedder.embed_batch(texts)
print(f'  Embedded {len(embeddings)} chunks ({settings.embedding_model}, dim={settings.embedding_dimension})')

# Step 5: Store in ChromaDB
print('Step 5: Storing in ChromaDB...')
# Import here to handle potential chromadb compatibility issues
from src.vectorstore.chroma_store import ChromaVectorStore

store = ChromaVectorStore(
    collection_name='rag_documents',
    persist_dir=settings.chroma_persist_dir,
)
ids = store.add_documents(chunks, embeddings)

# Print stats
stats = store.get_stats()
print('')
print('=== Seeding Complete ===')
print(f'  Documents loaded:  {len(all_docs)}')
print(f'  Chunks created:    {len(chunks)}')
print(f'  Embeddings stored: {len(ids)}')
print(f'  Collection:        {stats[\"collection_name\"]}')
print(f'  Persist dir:       {stats[\"persist_dir\"]}')
print(f'  Total in store:    {stats[\"total_documents\"]}')
"

echo ""
echo "Done."
