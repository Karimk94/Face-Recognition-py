"""
Migration Script — Seed ChromaDB Embeddings from known_faces_db
==============================================================
Run this **once** after deploying the Dataiku Model API to enroll all
existing faces from the local ``known_faces_db/`` directory.

The script reads each face image, Base64-encodes it, and calls Dataiku's
``/get_embeddings`` endpoint to compute the vector. It then stores the
vector in the local ChromaDB database.

Usage::

    # Set the Dataiku URL if not on localhost:5009
    set DATAIKU_MODEL_API_URL=http://your-dataiku-host:5009

    python migrate_embeddings.py
"""

import os
import sys
import base64
import requests
import uuid
import chromadb
from dotenv import load_dotenv

load_dotenv()

DATAIKU_API_URL = os.environ.get("DATAIKU_MODEL_API_URL", "http://localhost:5003")
KNOWN_FACES_PATH = os.path.abspath(
    os.environ.get("KNOWN_FACES_PATH", "known_faces_db")
)
CHROMADB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_db"))

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

    print(f"\nInitializing ChromaDB at '{CHROMADB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = chroma_client.get_or_create_collection(
        name="faces",
        metadata={"hnsw:space": "l2"}
    )

    print(f"Scanning '{KNOWN_FACES_PATH}'...\n")

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
                    f"{DATAIKU_API_URL.rstrip('/')}/get_embeddings",
                    json={"faces": [face_b64]},
                    timeout=120,
                )
                resp.raise_for_status()
                result = resp.json()
                embeddings = result.get("embeddings", [])
                
                if embeddings and embeddings[0]:
                    face_id = str(uuid.uuid4())
                    collection.add(
                        embeddings=[embeddings[0]],
                        metadatas=[{"name": person_folder}],
                        ids=[face_id]
                    )
                    print(f"    OK  {face_file} — enrolled")
                    total += 1
                else:
                    print(f"    ERR {face_file} — Dataiku returned no embedding")
                    errors += 1

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
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        counts = {}
        for meta in metadatas:
            name = meta.get("name", "Unknown")
            counts[name] = counts.get(name, 0) + 1
            
        print(f"\nChromaDB now has {len(metadatas)} embeddings for {len(counts)} people:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count} embedding(s)")
    except Exception as e:
        print(f"Failed to verify ChromaDB: {e}")


if __name__ == "__main__":
    migrate()
