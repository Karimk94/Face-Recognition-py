"""
database.py
===========
ChromaDB client and face-embedding collection singleton.
Import ``collection`` wherever you need to query or mutate the vector store.
"""

import logging
import chromadb
from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent ChromaDB client
# ---------------------------------------------------------------------------

chroma_client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)

try:
    collection = chroma_client.get_or_create_collection(
        name="faces",
        metadata={"hnsw:space": "l2"},  # VGG-Face works well with L2 distance
    )
except Exception as exc:
    logger.error("Failed to load chroma collection: %s", exc)
    raise
