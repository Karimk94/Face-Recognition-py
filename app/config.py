"""
config.py
=========
Centralised application settings loaded from environment variables (and .env).
All other modules import from here instead of calling os.environ directly.
"""

import os


class Settings:
    """Reads configuration once at import time."""

    DATAIKU_API_URL: str = os.environ.get(
        "DATAIKU_MODEL_API_URL", "http://localhost:5003"
    )
    REQUEST_TIMEOUT: int = int(os.environ.get("DATAIKU_TIMEOUT_SECONDS", 300))

    # ChromaDB persistence directory (same folder as the project root)
    CHROMADB_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "chroma_db"
    )


settings = Settings()
