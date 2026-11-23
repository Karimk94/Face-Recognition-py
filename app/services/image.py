"""
services/image.py
=================
Core image-processing pipeline.

Steps
-----
1. Forward the raw image to Dataiku for combined detection + embedding
   (get_faces_bbox_and_embeddings endpoint — single call, no crop upload).
2. Crop each detected face locally with PIL and generate ``thumbnail_b64``.
3. Query ChromaDB with the returned embeddings for recognition.
4. Draw bounding boxes + labels on the original image.
5. Return the same JSON structure the frontend expects.
"""

import base64
import io
import logging

from PIL import Image, ImageDraw, ImageFont

from app.database import collection
from app.services.dataiku import get_faces_bbox_and_embeddings
from app.config import settings

logger = logging.getLogger(__name__)


def process_image_via_dataiku(image_bytes: bytes) -> dict:
    """Run the full face detection + recognition pipeline.

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        A dict with keys ``faces``, ``processed_image``, and
        ``original_image_b64`` — identical to the original Flask response.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # ---- Step 1: Detect faces AND get embeddings in a single Dataiku call ----
    # Uses /get_faces_bbox_and_embeddings/run which accepts the full image and
    # returns both bounding boxes and 4096-dim VGG-Face vectors together.
    # This avoids the separate /get_face_embedding/run call that crops cause
    # issues with.
    try:
        payload = get_faces_bbox_and_embeddings(image_b64)
    except Exception as exc:
        logger.error("get_faces_bbox_and_embeddings failed: %s", exc, exc_info=True)
        payload = []

    # Normalise payload: expect a list of {bbox, confidence, embedding}
    if not isinstance(payload, list):
        payload = []

    logger.info(
        "get_faces_bbox_and_embeddings returned %d face(s).",
        len(payload),
    )

    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    results: dict = {"faces": []}

    # No faces detected — return the image untouched
    if not payload:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        results["processed_image"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        results["original_image_b64"] = image_b64
        return results

    # ---- Step 2: Crop faces locally and generate thumbnails ---------------
    crops_b64: list[str] = []
    detected_faces = []
    embeddings = []

    for item in payload:
        bbox = item.get("bbox", {})
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("w", 0))
        h = int(bbox.get("h", 0))
        emb = item.get("embedding")

        detected_faces.append({
            "x": x, "y": y, "w": w, "h": h,
            "confidence": item.get("confidence", 0),
        })
        embeddings.append(emb)

        cropped = image.crop((x, y, x + w, y + h))
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG")
        crops_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    # ---- Step 3: Query ChromaDB with the embeddings -----------------------
    recognized: list[dict] = []
    if collection.count() > 0:
        for emb in embeddings:
            if emb is None:
                recognized.append({"name": "Unknown", "distance": None})
                continue
            try:
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
                    # Only accept the match if it is within the threshold.
                    # Distances above the threshold are too dissimilar → Unknown.
                    if distance <= settings.FACE_MATCH_THRESHOLD:
                        recognized.append({"name": name, "distance": distance})
                    else:
                        logger.debug(
                            "Face rejected: best match '%s' distance %.4f > threshold %.4f",
                            name, distance, settings.FACE_MATCH_THRESHOLD,
                        )
                        recognized.append({"name": "Unknown", "distance": distance})
                else:
                    recognized.append({"name": "Unknown", "distance": None})
            except Exception as exc:
                logger.error("ChromaDB query failed: %s", exc, exc_info=True)
                recognized.append({"name": "Unknown", "distance": None})

    # ---- Step 4: Draw bounding boxes and merge results --------------------
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for i, face in enumerate(detected_faces):
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]

        name = "Unknown"
        distance = None
        if i < len(recognized):
            name = recognized[i].get("name", "Unknown")
            distance = recognized[i].get("distance")

        draw.rectangle(((x, y), (x + w, y + h)), outline=(0, 255, 0), width=3)
        face_number = i + 1
        label_text = f"#{face_number}: {name.replace('_', ' ').title()}"

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        draw.text((x, y - 20), label_text, font=font, fill=(0, 255, 0))

        results["faces"].append(
            {
                "index": face_number,
                "name": name,
                "location": [y, x + w, y + h, x],  # top, right, bottom, left
                "distance": distance,
                "thumbnail_b64": crops_b64[i] if i < len(crops_b64) else None,
            }
        )

    buf = io.BytesIO()
    draw_image.save(buf, format="JPEG")
    results["processed_image"] = base64.b64encode(buf.getvalue()).decode("utf-8")
    results["original_image_b64"] = image_b64

    return results
