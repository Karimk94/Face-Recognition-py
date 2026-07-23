"""
services/dataiku.py
===================
HTTP helper for the Dataiku Model API.
All other modules call ``call_dataiku`` rather than building raw HTTP requests.
"""

import requests as http_client
from app.config import settings


def call_dataiku(endpoint: str, json_data: dict) -> dict:
    """POST JSON to the Dataiku Model API and return the parsed response.

    Args:
        endpoint: Path relative to the base URL, e.g. ``"/detect_faces"``.
        json_data: Payload to send as JSON.

    Returns:
        Parsed JSON response body as a dict.

    Raises:
        requests.HTTPError: If the upstream API returns a non-2xx status.
    """
    url = f"{settings.DATAIKU_API_URL.rstrip('/')}{endpoint}"
    response = http_client.post(url, json=json_data, timeout=settings.REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()
