"""
test_dataiku.py — Diagnostic: call the Dataiku Model API directly
=================================================================
Run this ON THE SERVER (or any machine that can resolve the Dataiku host)
to inspect the RAW responses from the Model API and verify the JSON schema
matches what the gateway expects.

Usage::

    venv\\Scripts\\python.exe test_dataiku.py path\\to\\photo.jpg

What it checks
--------------
1. POST /public/api/v1/edms-fr-test/detect_faces/run
   -> gateway expects:  {"faces": [{"x":..,"y":..,"w":..,"h":..}, ...]}
2. POST /public/api/v1/edms-fr-test/get_faces_bbox_and_embeddings/run
   -> combined detection + embeddings
3. POST /public/api/v1/edms-fr-test/get_face_embedding/run
   -> gateway expects:  {"embeddings": [[...floats...]]}

Every endpoint is wrapped in its own try/except — one failure never
aborts the whole diagnostic.
"""

import base64
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get(
    "DATAIKU_MODEL_API_URL", "https://rtauevaidss1.rtatestdom.local:13000"
).rstrip("/")
# Diagnostic script: default to NOT verifying (internal self-signed cert),
# so it also works from a plain cmd session where web.config env vars
# (injected by IIS only into the worker process) are not present.
VERIFY_SSL = os.environ.get("DATAIKU_VERIFY_SSL", "false").strip().lower() != "false"
TIMEOUT = int(os.environ.get("DATAIKU_TIMEOUT_SECONDS", 300))

if not VERIFY_SSL:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = f"{BASE_URL}/public/api/v1/edms-fr-test"


def _embedding_dims(payload) -> None:
    """Print the dimension of any embedding vectors found in a payload."""
    items = payload if isinstance(payload, list) else [payload]
    for i, item in enumerate(items):
        vec = None
        if isinstance(item, dict):
            for key in ("embedding", "vector"):
                if isinstance(item.get(key), list):
                    vec = item[key]
                    break
        elif isinstance(item, list) and item and isinstance(item[0], (int, float)):
            vec = item
        if vec is not None:
            print(f"  >> embedding[{i}] dimension: {len(vec)}")


def call(title: str, url: str, payload: dict) -> None:
    """POST to one endpoint and print the result — never raises."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    try:
        resp = requests.post(url, json=payload, timeout=TIMEOUT, verify=VERIFY_SSL)
    except requests.exceptions.SSLError as exc:
        print(f"SSL ERROR — certificate problem: {exc}")
        print("Hint: run 'set DATAIKU_VERIFY_SSL=false' first, then retry.")
        return
    except requests.exceptions.ConnectionError as exc:
        print(f"CONNECTION ERROR — host unreachable or unresolvable: {exc}")
        print("Hint: this machine cannot reach the Dataiku host. Run on the server.")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        return

    print(f"HTTP {resp.status_code}")
    try:
        body = resp.json()
        # Report embedding dimensions — the stored ChromaDB expects 4096-dim
        # vectors; a mismatch means chroma_db must be regenerated on the server.
        inner = body.get("response", body) if isinstance(body, dict) else body
        if isinstance(inner, dict) and "embeddings" in inner:
            _embedding_dims(inner["embeddings"])
        else:
            _embedding_dims(inner)
        text = json.dumps(body, indent=2)
        # Truncate huge values (embeddings / base64) so the shape stays visible
        print(text[:2000])
        if len(text) > 2000:
            print(f"... <truncated, total {len(text)} chars>")
    except ValueError:
        print(resp.text[:2000])


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_dataiku.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"ERROR: file not found: {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    print("test_dataiku.py v2")
    print(f"Target API : {BASE_URL}")
    print(f"Verify SSL : {VERIFY_SSL}")
    print(f"Image      : {image_path} ({os.path.getsize(image_path)} bytes)")

    call("1) detect_faces/run", f"{API_BASE}/detect_faces/run",
         {"image_b64": image_b64})

    call("2) get_faces_bbox_and_embeddings/run",
         f"{API_BASE}/get_faces_bbox_and_embeddings/run",
         {"image_b64": image_b64})

    call("3) get_face_embedding/run", f"{API_BASE}/get_face_embedding/run",
         {"images_b64": [image_b64]})


if __name__ == "__main__":
    main()
