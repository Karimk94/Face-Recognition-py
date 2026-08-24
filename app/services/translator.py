"""
services/translator.py
======================
Thin wrapper around the Translator / Rephraser API.

Used to normalise face names that arrive in Arabic (when the UI is in Arabic
mode) into their English equivalents before storing them in ChromaDB.

The API streams SSE-style lines:
    data: <chunk>
    data: [END_OF_STREAM]
"""

import logging

import requests

from app.config import settings

logger = logging.getLogger(__name__)


def translate_to_english(text: str) -> str | None:
    """Translate *text* to English via the Translator/Rephraser service.

    Returns the translated string on success, or ``None`` if the service is
    unavailable or the call fails (so the caller can decide what to do).
    """
    if not settings.TRANSLATOR_API_URL:
        logger.warning("TRANSLATOR_REPHRASER_API_URL is not set; cannot translate name.")
        return None

    try:
        response = requests.post(
            f"{settings.TRANSLATOR_API_URL.rstrip('/')}/generate",
            json={"text": text, "task": "translate"},
            timeout=60,
            stream=True,
        )
        response.raise_for_status()

        result = ""
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    chunk = decoded[6:]
                    if chunk.strip() == "[END_OF_STREAM]":
                        break
                    result += chunk

        return result.strip() or None

    except Exception as exc:
        logger.error("Translation request failed: %s", exc, exc_info=True)
        return None
