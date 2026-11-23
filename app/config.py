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

    # Retry configuration for Dataiku API calls
    DATAIKU_MAX_RETRIES: int = int(os.environ.get("DATAIKU_MAX_RETRIES", 3))
    DATAIKU_RETRY_DELAY: float = float(os.environ.get("DATAIKU_RETRY_DELAY_SECONDS", 1.0))
    DATAIKU_RETRY_BACKOFF: float = float(os.environ.get("DATAIKU_RETRY_BACKOFF_MULTIPLIER", 2.0))

    # Face recognition distance threshold (VGG-Face + L2 distance).
    # Matches with a distance above this value are treated as "Unknown".
    # DeepFace default is 0.86 for VGG-Face L2. Lower = stricter matching.
    # Configurable via FACE_MATCH_THRESHOLD env var.
    FACE_MATCH_THRESHOLD: float = float(os.environ.get("FACE_MATCH_THRESHOLD", 0.86))

    # Translator / Rephraser — now calls GovAI directly (no microservice dependency)
    GOVAI_API_URL: str = os.environ.get("GOVAI_API_URL", "https://llmapi.govai.ae/chat/completions")
    GOVAI_API_KEY: str = os.environ.get("GOVAI_API_KEY", "")
    GOVAI_TEXT_MODEL: str = os.environ.get("GOVAI_TEXT_MODEL", "qwen3-235b-fp16")


settings = Settings()
