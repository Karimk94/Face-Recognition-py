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

def _post(endpoint: str, json_data: dict) -> dict:
    """POST JSON to the Model API and return the parsed response.

    Args:
        endpoint: Full path relative to the base URL.
        json_data: Payload to send as JSON.

    Returns:
        Parsed JSON response body as a dict (envelope still wrapped).

    Raises:
        requests.HTTPError: If the upstream API returns a non-2xx status.
    """
    url = f"{settings.DATAIKU_API_URL.rstrip('/')}{endpoint}"
    response = http_client.post(
        url,
        json=json_data,
        timeout=settings.REQUEST_TIMEOUT,
        verify=settings.VERIFY_SSL,
    )
    response.raise_for_status()
    return response.json()


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
    """
    embeddings: list = []
    for start in range(0, len(images_b64), _BATCH_SIZE):
        batch = images_b64[start:start + _BATCH_SIZE]
        payload = _unwrap_envelope(
            _post(ENDPOINT_GET_FACE_EMBEDDING, {"images_b64": batch})
        )
        embeddings.extend(_normalize_embeddings_payload(payload))
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
