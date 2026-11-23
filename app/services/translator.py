"""
services/translator.py
======================
Translates face names from Arabic to English using the GovAI LLM API directly.

Previously this was a thin wrapper around the Translator/Rephraser microservice
(:5005). After MCP migration, it calls GovAI chat/completions directly with SSE
streaming — same model (qwen3-235b-fp16), same prompts, same post-processing.

Used to normalise face names that arrive in Arabic (when the UI is in Arabic
mode) into their English equivalents before storing them in ChromaDB.
"""

import json
import logging
import re

import requests

from app.config import settings

logger = logging.getLogger(__name__)


def _is_arabic(text: str) -> bool:
    """Detects if the text contains Arabic characters."""
    if not text or not isinstance(text, str):
        return False
    return bool(re.search(r'[\u0600-\u06FF]', text))


def _clean_response(text: str) -> str:
    """Removes Arabic diacritics (Tashkeel) and stray question marks."""
    if not text:
        return text
    arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    cleaned = re.sub(arabic_diacritics, '', text)
    cleaned = cleaned.replace('?', '')
    return cleaned


def translate_to_english(text: str) -> str | None:
    """Translate *text* to English via the GovAI LLM API.

    Returns the translated string on success, or ``None`` if the API is
    unavailable or the call fails (so the caller can decide what to do).
    """
    if not settings.GOVAI_API_URL or not settings.GOVAI_API_KEY:
        logger.warning("GOVAI_API_URL/GOVAI_API_KEY not set; cannot translate name.")
        return None

    # If text is not Arabic, no translation needed
    if not _is_arabic(text):
        return text

    source_lang = "Arabic"
    target_lang = "English"

    system_msg = f"You are a direct translation engine. Translate the provided {source_lang} text to {target_lang}."
    user_msg = f"""Follow these rules exactly:
1. Your response MUST contain ONLY the translated text.
2. Do NOT add any comments.
3. **CRITICAL**: You MUST convert every single {source_lang} word into {target_lang}.

### TASK ###
{source_lang} Text: "{text}" """

    payload = {
        "model": settings.GOVAI_TEXT_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0.3,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.GOVAI_API_KEY}",
    }

    try:
        response = requests.post(
            settings.GOVAI_API_URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

        result = ""
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data_str = decoded[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                result += _clean_response(content)
                    except json.JSONDecodeError:
                        continue

        return result.strip() or None

    except Exception as exc:
        logger.error("GovAI translation request failed: %s", exc, exc_info=True)
        return None
