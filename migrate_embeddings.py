"""
Migration Script — Seed Dataiku Embeddings from known_faces_db
==============================================================
Run this **once** after deploying the Dataiku Model API to enroll all
existing faces from the local ``known_faces_db/`` directory.

The script reads each face image, Base64-encodes it, and calls Dataiku's
``/enroll_face`` endpoint.  Only the embedding vector is stored on Dataiku —
the actual image stays here on the Thumbnail API server.

Usage::

    # Set the Dataiku URL if not on localhost:5003
    set DATAIKU_MODEL_API_URL=http://your-dataiku-host:5003

    python migrate_embeddings.py
"""

import os
import sys
import base64
import requests

DATAIKU_API_URL = os.environ.get("DATAIKU_MODEL_API_URL", "http://localhost:5003")
KNOWN_FACES_PATH = os.path.abspath(
    os.environ.get("KNOWN_FACES_PATH", "known_faces_db")
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def migrate():
    if not os.path.exists(KNOWN_FACES_PATH):
        print(f"ERROR: '{KNOWN_FACES_PATH}' does not exist.")
        sys.exit(1)

    # Quick health check
    try:
        resp = requests.get(f"{DATAIKU_API_URL.rstrip('/')}/health", timeout=10)
        resp.raise_for_status()
        health = resp.json()
        print(f"Dataiku Model API is healthy: {health}")
    except Exception as e:
        print(f"ERROR: Cannot reach Dataiku Model API at {DATAIKU_API_URL}: {e}")
        sys.exit(1)

    print(f"\nScanning '{KNOWN_FACES_PATH}'...\n")

    total = 0
    errors = 0
    skipped = 0

    for person_folder in sorted(os.listdir(KNOWN_FACES_PATH)):
        person_path = os.path.join(KNOWN_FACES_PATH, person_folder)
        if not os.path.isdir(person_path):
            continue

        print(f"  [{person_folder}]")

        for face_file in sorted(os.listdir(person_path)):
            ext = os.path.splitext(face_file)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                skipped += 1
                continue

            face_path = os.path.join(person_path, face_file)
            try:
                with open(face_path, "rb") as f:
                    face_b64 = base64.b64encode(f.read()).decode("utf-8")

                resp = requests.post(
                    f"{DATAIKU_API_URL.rstrip('/')}/enroll_face",
                    json={"name": person_folder, "face_b64": face_b64},
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                print(f"    OK  {face_file} — {result.get('message', 'enrolled')}")
                total += 1

            except Exception as e:
                print(f"    ERR {face_file} — {e}")
                errors += 1

    print(f"\n{'='*50}")
    print(f"Migration complete:")
    print(f"  Enrolled : {total}")
    print(f"  Errors   : {errors}")
    print(f"  Skipped  : {skipped} (non-image files)")
    print(f"{'='*50}")

    # Final check
    try:
        resp = requests.get(f"{DATAIKU_API_URL.rstrip('/')}/list_faces", timeout=10)
        resp.raise_for_status()
        faces = resp.json().get("faces", {})
        print(f"\nDataiku now has {sum(faces.values())} embeddings for {len(faces)} people:")
        for name, count in sorted(faces.items()):
            print(f"  {name}: {count} embedding(s)")
    except Exception:
        pass


if __name__ == "__main__":
    migrate()
