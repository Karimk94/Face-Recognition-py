"""
config.py
=========
Centralised application settings loaded from environment variables (and .env).
All other modules import from here instead of calling os.environ directly.
"""

import os

from dotenv import load_dotenv

# Load .env here too, so settings stay correct even when the app is started
# directly with `uvicorn app.main:app` (which bypasses the app.py shim).
load_dotenv()


class Settings:
    """Reads configuration once at import time."""

    DATAIKU_API_URL: str = os.environ.get(
        "DATAIKU_MODEL_API_URL", "https://rtauevaidss1.rtatestdom.local:13000"
    )
    REQUEST_TIMEOUT: int = int(os.environ.get("DATAIKU_TIMEOUT_SECONDS", 300))

    # SSL certificate verification for calls to the Dataiku Model API.
    # Set DATAIKU_VERIFY_SSL="false" for internal test domains that use a
    # self-signed / internal-CA certificate.
    VERIFY_SSL: bool = os.environ.get("DATAIKU_VERIFY_SSL", "true").strip().lower() != "false"

    # ChromaDB persistence directory (same folder as the project root)
    CHROMADB_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "chroma_db"
    )


settings = Settings()
