"""
services/dataiku.py
===================
HTTP helper for the Dataiku Model API.
All other modules call the helpers below rather than building raw HTTP requests.

API base: configured via DATAIKU_MODEL_API_URL env var.

Envelope
--------
The Dataiku API Deployer wraps EVERY endpoint result as::

    {"response": <payload>, "timing": {...}, "apiContext": {...}}

The helpers below unwrap that envelope and normalise the payload so the rest
of the app never has to care about the wire format.

Endpoints (observed payloads, after unwrapping)
-----------------------------------------------
POST /public/api/v1/edms-fr-test/detect_faces/run
    Body: {"image_b64": "<single image>"}
    Payload: [{"bbox": {"x","y","w","h"}, "confidence": 1.0}, ...]
    Normalised to: {"faces": [{"x","y","w","h","confidence"}, ...]}

POST /public/api/v1/edms-fr-test/get_face_embedding/run
    Body: {"images_b64": ["<img1>", ...]}   (up to 5 images)
    Payload: [[...vector...], ...]  (or [{"embedding": [...]}, ...])
    Normalised to: {"embeddings": [...]}

POST /public/api/v1/edms-fr-test/get_faces_bbox_and_embeddings/run
    Body: {"image_b64": "<single image>"}
    Returns combined detection + embeddings in one call
"""

import logging
import time
from typing import Optional

import requests as http_client
from app.config import settings

logger = logging.getLogger(__name__)

if not settings.VERIFY_SSL:
    # Calling an internal test domain with a self-signed / internal-CA cert.
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_BASE = "/public/api/v1/edms-fr-test"

ENDPOINT_DETECT_FACES          = f"{_BASE}/detect_faces/run"
ENDPOINT_GET_FACE_EMBEDDING    = f"{_BASE}/get_face_embedding/run"
ENDPOINT_BBOX_AND_EMBEDDINGS   = f"{_BASE}/get_faces_bbox_and_embeddings/run"

# The Model API accepts at most 5 images per get_face_embedding call.
# get_face_embeddings() below transparently splits larger lists into batches.
_BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, json_data: dict, max_retries: Optional[int] = None) -> dict:
    """POST JSON to the Model API and return the parsed response.

    Args:
        endpoint: Full path relative to the base URL.
        json_data: Payload to send as JSON.
        max_retries: Maximum number of retry attempts for transient failures.
                     Defaults to settings.DATAIKU_MAX_RETRIES.

    Returns:
        Parsed JSON response body as a dict (envelope still wrapped).

    Raises:
        requests.HTTPError: If the upstream API returns a non-2xx status after all retries.
    """
    if max_retries is None:
        max_retries = settings.DATAIKU_MAX_RETRIES
    
    url = f"{settings.DATAIKU_API_URL.rstrip('/')}{endpoint}"
    last_exception: Optional[Exception] = None
    delay = settings.DATAIKU_RETRY_DELAY

    for attempt in range(max_retries + 1):
        try:
            response = http_client.post(
                url,
                json=json_data,
                timeout=settings.REQUEST_TIMEOUT,
                verify=settings.VERIFY_SSL,
            )
            response.raise_for_status()
            return response.json()

        except http_client.HTTPError as exc:
            last_exception = exc
            status_code = exc.response.status_code if exc.response else "N/A"
            
            # Try to capture response body for debugging
            response_body = "Unable to read response body"
            try:
                response_body = exc.response.text[:1000] if exc.response else "No response"
            except Exception:
                pass

            # Retry on 5xx server errors (transient failures)
            if isinstance(status_code, int) and 500 <= status_code < 600:
                if attempt < max_retries:
                    logger.warning(
                        "API request failed with %s (attempt %d/%d). Retrying in %.1fs. "
                        "URL: %s, Response: %s",
                        status_code,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                        endpoint,
                        response_body,
                    )
                    time.sleep(delay)
                    delay *= settings.DATAIKU_RETRY_BACKOFF
                    continue
                else:
                    logger.error(
                        "API request failed after %d attempts. Status: %s, URL: %s, Response: %s",
                        max_retries + 1,
                        status_code,
                        endpoint,
                        response_body,
                    )
            # For 4xx client errors, don't retry - log and re-raise immediately
            elif isinstance(status_code, int) and 400 <= status_code < 500:
                logger.error(
                    "Client error %s for API request. URL: %s, Response: %s",
                    status_code,
                    endpoint,
                    response_body,
                )
            raise

        except http_client.Timeout as exc:
            last_exception = exc
            if attempt < max_retries:
                logger.warning(
                    "API request timed out (attempt %d/%d). Retrying in %.1fs. URL: %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    endpoint,
                )
                time.sleep(delay)
                delay *= settings.DATAIKU_RETRY_BACKOFF
                continue
            else:
                logger.error(
                    "API request timed out after %d attempts. URL: %s",
                    max_retries + 1,
                    endpoint,
                )
            raise

        except http_client.ConnectionError as exc:
            last_exception = exc
            if attempt < max_retries:
                logger.warning(
                    "Connection error (attempt %d/%d). Retrying in %.1fs. URL: %s, Error: %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    endpoint,
                    str(exc),
                )
                time.sleep(delay)
                delay *= settings.DATAIKU_RETRY_BACKOFF
                continue
            else:
                logger.error(
                    "Connection error after %d attempts. URL: %s, Error: %s",
                    max_retries + 1,
                    endpoint,
                    str(exc),
                )
            raise

    # Should not reach here, but raise the last exception if we do
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in _post retry logic")


def _unwrap_envelope(body):
    """Unwrap the Dataiku API Deployer envelope.

    ``{"response": <payload>, "timing": ..., "apiContext": ...}`` -> ``<payload>``
    Anything that is not a dict with a ``"response"`` key is returned unchanged.
    """
    if isinstance(body, dict) and "response" in body:
        return body["response"]
    return body


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_faces(image_b64: str) -> dict:
    """Detect faces (bounding boxes) in a single image.

    Args:
        image_b64: Base64-encoded image string.

    Returns:
        ``{"faces": [{"x": .., "y": .., "w": .., "h": .., "confidence": ..}, ...]}``
    """
    payload = _unwrap_envelope(_post(ENDPOINT_DETECT_FACES, {"image_b64": image_b64}))

    # Shape A (observed on Dataiku): [{"bbox": {...}, "confidence": 1.0}, ...]
    if isinstance(payload, list):
        faces = []
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("bbox"), dict):
                bbox = item["bbox"]
                face = {k: bbox[k] for k in ("x", "y", "w", "h") if k in bbox}
                if "confidence" in item:
                    face["confidence"] = item["confidence"]
                faces.append(face)
            elif isinstance(item, dict) and {"x", "y", "w", "h"} <= item.keys():
                faces.append(item)
        return {"faces": faces}

    # Shape B (legacy/local mock): {"faces": [...]}
    if isinstance(payload, dict):
        return payload

    logger.warning(
        "Unexpected detect_faces payload type: %s", type(payload).__name__
    )
    return {"faces": []}


def _normalize_embeddings_payload(payload) -> list:
    """Normalise one get_face_embedding payload to a plain list of vectors."""
    # Shape A: {"embeddings": [...]} (legacy/local, or dict payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("embeddings"), list):
            return payload["embeddings"]
        logger.warning(
            "Unexpected get_face_embedding payload keys: %s", list(payload.keys())
        )
        return []

    # Shape B: [ [...vector...], {"embedding": [...]}, None, ... ]
    if isinstance(payload, list):
        embeddings = []
        for item in payload:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append(item["embedding"])
            else:
                embeddings.append(item)  # plain vector or None
        return embeddings

    logger.warning(
        "Unexpected get_face_embedding payload type: %s", type(payload).__name__
    )
    return []


def get_face_embeddings(images_b64: list[str]) -> dict:
    """Compute face embeddings for a list of pre-cropped face images.

    The Model API accepts **at most 5 images per call** — larger lists are
    automatically split into batches of 5 and the results concatenated in
    order, so callers never need to worry about the limit.

    Args:
        images_b64: List of base64-encoded face crop strings (any length).

    Returns:
        ``{"embeddings": [...]}`` — one entry per input image, in the same
        order (``None`` where no face/vector could be computed).

    Raises:
        requests.HTTPError: If the API fails after all retry attempts.
    """
    embeddings: list = []
    total_images = len(images_b64)
    
    for start in range(0, total_images, _BATCH_SIZE):
        batch = images_b64[start:start + _BATCH_SIZE]
        batch_num = (start // _BATCH_SIZE) + 1
        total_batches = (total_images + _BATCH_SIZE - 1) // _BATCH_SIZE
        
        logger.debug(
            "Processing embedding batch %d/%d (%d images)",
            batch_num,
            total_batches,
            len(batch),
        )
        
        try:
            payload = _unwrap_envelope(
                _post(ENDPOINT_GET_FACE_EMBEDDING, {"images_b64": batch})
            )
            batch_embeddings = _normalize_embeddings_payload(payload)
            embeddings.extend(batch_embeddings)
        except Exception as exc:
            logger.error(
                "Failed to get embeddings for batch %d/%d (%d images): %s",
                batch_num,
                total_batches,
                len(batch),
                exc,
            )
            # Re-raise to let the caller handle the failure
            raise

    return {"embeddings": embeddings}


def get_faces_bbox_and_embeddings(image_b64: str):
    """Detect faces AND compute their embeddings in a single API call.

    Args:
        image_b64: Base64-encoded image string.

    Returns:
        The unwrapped payload (dict or list, depending on the Model API).
    """
    return _unwrap_envelope(_post(ENDPOINT_BBOX_AND_EMBEDDINGS, {"image_b64": image_b64}))


# ---------------------------------------------------------------------------
# Backwards-compat shim (used by older call sites that pass an endpoint str)
# ---------------------------------------------------------------------------

def call_dataiku(endpoint: str, json_data: dict) -> dict:
    """Generic POST — returns the RAW body (envelope NOT unwrapped)."""
    return _post(endpoint, json_data)
