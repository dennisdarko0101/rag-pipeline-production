#!/usr/bin/env bash
set -euo pipefail

echo "Seeding vector database with sample documents..."
python3 -c "
from src.config.settings import settings
print(f'Using ChromaDB at: {settings.chroma_persist_dir}')
print('Seed script will be implemented in Phase 1, Step 3')
"
echo "Done."
