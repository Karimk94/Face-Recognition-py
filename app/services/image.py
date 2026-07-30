"""
services/image.py
=================
Core image-processing pipeline.

Steps
-----
1. Forward the raw image to Dataiku for face detection (bounding boxes only).
2. Crop each detected face locally with PIL and generate ``thumbnail_b64``.
3. Send cropped thumbnails to Dataiku to compute embeddings, then query
   ChromaDB for recognition.
4. Draw bounding boxes + labels on the original image.
5. Return the same JSON structure the frontend expects.
"""

import base64
import io
import logging

from PIL import Image, ImageDraw, ImageFont

from app.database import collection
from app.services.dataiku import detect_faces, get_face_embeddings

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

    # ---- Step 1: Detect faces via the Model API --------------------------------
    detection = detect_faces(image_b64)
    detected_faces = detection.get("faces", [])

    # Debug: surface the raw Model API response in the logs so deployment
    # issues (schema mismatch, empty detections) are easy to diagnose.
    logger.info(
        "detect_faces returned %d face(s). Raw response (truncated): %s",
        len(detected_faces) if isinstance(detected_faces, list) else "N/A",
        str(detection)[:500],
    )

    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    results: dict = {"faces": []}

    # No faces detected — return the image untouched
    if not detected_faces:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        results["processed_image"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        results["original_image_b64"] = image_b64
        return results

    # ---- Step 2: Crop faces locally and generate thumbnails ---------------
    crops_b64: list[str] = []
    for face in detected_faces:
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        cropped = image.crop((x, y, x + w, y + h))
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG")
        crops_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    # ---- Step 3: Compute embeddings & recognise via ChromaDB --------------
    recognized: list[dict] = []
    if crops_b64:
        try:
            response = get_face_embeddings(crops_b64)
            embeddings = response.get("embeddings", [])

            for emb in embeddings:
                if emb is None:
                    recognized.append({"name": "Unknown", "distance": None})
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
                    recognized.append({"name": name, "distance": distance})
                else:
                    recognized.append({"name": "Unknown", "distance": None})

        except Exception as exc:
            logger.error("Recognition step failed: %s", exc, exc_info=True)

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
