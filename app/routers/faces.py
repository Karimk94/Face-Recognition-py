"""
routers/faces.py
================
Face management endpoints:

  POST /api/add_face        — Enrol a new face into ChromaDB
  POST /api/recognize_faces — Recognise a list of pre-cropped base64 faces
  POST /api/delete_face     — Remove all embeddings for a person
  GET  /api/list_faces      — List all enrolled people and embedding counts
"""

import base64
import io
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from app.database import collection
from app.services.dataiku import get_face_embeddings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["faces"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class AddFaceRequest(BaseModel):
    name: str
    location: list[int]          # [top, right, bottom, left]
    original_image_b64: str


class RecognizeFacesRequest(BaseModel):
    faces: list[str]             # list of base64-encoded face crops


class DeleteFaceRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/add_face")
async def api_add_face(body: AddFaceRequest) -> Any:
    """Enrol a new face.

    1. Crop the face from the original image using the provided location.
    2. Send the crop to Dataiku to compute the embedding.
    3. Store the embedding in ChromaDB.
    """
    try:
        image_bytes = base64.b64decode(body.original_image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        top, right, bottom, left = body.location
        cropped_face = image.crop((left, top, right, bottom)).convert("RGB")

        buf = io.BytesIO()
        cropped_face.save(buf, format="JPEG")
        face_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            response = get_face_embeddings([face_b64])
            embeddings = response.get("embeddings", [])

            if embeddings and embeddings[0]:
                face_id = str(uuid.uuid4())
                collection.add(
                    embeddings=[embeddings[0]],
                    metadatas=[{"name": body.name.replace(" ", "_").lower()}],
                    ids=[face_id],
                )
            else:
                logger.error("Dataiku did not return a valid embedding.")

        except Exception as exc:
            logger.error(
                "Saved face locally but ChromaDB insert failed: %s", exc, exc_info=True
            )

        return JSONResponse(content={"message": f"Saved '{body.name}' successfully."})

    except Exception as exc:
        logger.error("Error in /add_face: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {exc}") from exc


@router.post("/recognize_faces")
async def api_recognize_faces(body: RecognizeFacesRequest) -> Any:
    """Recognise pre-cropped faces supplied as base64 strings."""
    try:
        response = get_face_embeddings(body.faces)
        embeddings = response.get("embeddings", [])

        results = []
        for i, emb in enumerate(embeddings):
            if emb is None:
                results.append({"index": i + 1, "name": "Unknown", "distance": None})
                continue

            query_results = collection.query(
                query_embeddings=[emb],
                n_results=1,
            )

            if (
                query_results
                and query_results["ids"]
                and len(query_results["ids"][0]) > 0
            ):
                distance = query_results["distances"][0][0]
                metadata = query_results["metadatas"][0][0]
                name = metadata.get("name", "Unknown")
                results.append({"index": i + 1, "name": name, "distance": distance})
            else:
                results.append({"index": i + 1, "name": "Unknown", "distance": None})

        return JSONResponse(content={"faces": results})

    except Exception as exc:
        logger.error("Error in /recognize_faces: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {exc}") from exc


@router.post("/delete_face")
async def api_delete_face(body: DeleteFaceRequest) -> Any:
    """Delete all embeddings for a person from ChromaDB."""
    try:
        formatted_name = body.name.replace(" ", "_").lower()
        collection.delete(where={"name": formatted_name})
        return JSONResponse(
            content={"success": True, "message": f"Deleted '{body.name}' successfully."}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Server error: {exc}") from exc


@router.get("/list_faces")
async def api_list_faces() -> Any:
    """List all enrolled people and their embedding counts."""
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        counts: dict[str, int] = {}
        for meta in metadatas:
            name = meta.get("name", "Unknown")
            counts[name] = counts.get(name, 0) + 1

        return JSONResponse(content={"faces": counts})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Server error: {exc}") from exc
