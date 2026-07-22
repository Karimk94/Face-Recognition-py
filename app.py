"""
Thumbnail API (Face Recognition Gateway)
=========================================
A lightweight Flask API that serves as the frontend gateway for face analysis.
This API handles all image manipulation (cropping, thumbnails, bounding boxes)
and manages the ``known_faces_db`` directory on disk.

All ML inference (detection, recognition, embedding computation) is delegated
to the Dataiku Model API via HTTP.

Endpoints (unchanged from the frontend's perspective)
-----------------------------------------------------
POST /api/analyze_image          — Multipart form upload
POST /api/analyze_image_stream   — Raw byte stream
POST /api/add_face               — JSON with base64 image
POST /api/recognize_faces        — JSON with base64 face crops
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import os
import requests as http_client
from waitress import serve

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DATAIKU_API_URL = os.environ.get("DATAIKU_MODEL_API_URL", "http://localhost:5003")
KNOWN_FACES_PATH = os.path.abspath(
    os.environ.get("KNOWN_FACES_PATH", "known_faces_db")
)
os.makedirs(KNOWN_FACES_PATH, exist_ok=True)

REQUEST_TIMEOUT = int(os.environ.get("DATAIKU_TIMEOUT_SECONDS", 300))


# ---------------------------------------------------------------------------
# Dataiku HTTP helpers
# ---------------------------------------------------------------------------

def call_dataiku(endpoint, json_data):
    """POST JSON to the Dataiku Model API and return the parsed response."""
    url = f"{DATAIKU_API_URL.rstrip('/')}{endpoint}"
    response = http_client.post(url, json=json_data, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Core image processing pipeline
# ---------------------------------------------------------------------------

def process_image_via_dataiku(image_bytes):
    """
    Full pipeline:
    1. Forward image to Dataiku for face detection (coordinates only).
    2. Crop each face locally using PIL → generate ``thumbnail_b64``.
    3. Forward cropped face thumbnails to Dataiku for recognition.
    4. Draw bounding boxes + labels → generate ``processed_image``.
    5. Return the same JSON structure the frontend expects.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # ---- Step 1: Detect faces via Dataiku --------------------------------
    detection = call_dataiku("/detect_faces", {"image_b64": image_b64})
    detected_faces = detection.get("faces", [])

    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")

    results = {"faces": []}

    # No faces detected — return the image untouched
    if not detected_faces:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        results["processed_image"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        results["original_image_b64"] = image_b64
        return results

    # ---- Step 2: Crop faces locally and generate thumbnails ---------------
    crops_b64 = []
    for face in detected_faces:
        x, y, w, h = face["x"], face["y"], face["w"], face["h"]
        cropped = image.crop((x, y, x + w, y + h))
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG")
        crops_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

    # ---- Step 3: Recognize faces via Dataiku ------------------------------
    try:
        recognition = call_dataiku("/recognize_faces", {"faces": crops_b64})
        recognized = recognition.get("faces", [])
    except Exception as e:
        app.logger.error(f"Recognition step failed: {e}", exc_info=True)
        recognized = []

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

        # Draw bounding box
        draw.rectangle(((x, y), (x + w, y + h)), outline=(0, 255, 0), width=3)
        face_number = i + 1
        label_text = f"#{face_number}: {name.replace('_', ' ').title()}"

        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        draw.text((x, y - 20), label_text, font=font, fill=(0, 255, 0))

        results["faces"].append({
            "index": face_number,
            "name": name,
            "location": [y, x + w, y + h, x],  # top, right, bottom, left
            "distance": distance,
            "thumbnail_b64": crops_b64[i] if i < len(crops_b64) else None,
        })

    # Generate processed image with bounding boxes
    buf = io.BytesIO()
    draw_image.save(buf, format="JPEG")
    results["processed_image"] = base64.b64encode(buf.getvalue()).decode("utf-8")
    results["original_image_b64"] = image_b64

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/analyze_image", methods=["POST"])
def api_analyze_image():
    """Process an image uploaded as a multipart form file."""
    if "image_file" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image_file"]
    image_bytes = file.read()
    try:
        results = process_image_via_dataiku(image_bytes)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error in /analyze_image: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/analyze_image_stream", methods=["POST"])
def analyze_image_stream():
    """Process a single image sent as a raw byte stream."""
    try:
        image_bytes = request.get_data()
        if not image_bytes:
            return jsonify(error="No image data in request body"), 400

        results = process_image_via_dataiku(image_bytes)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error in /analyze_image_stream: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/add_face", methods=["POST"])
def api_add_face():
    """
    Enroll a new face:
    1. Crop the face from the original image.
    2. Save the cropped image locally in ``known_faces_db/``.
    3. Send the crop to Dataiku to compute and store the embedding.
    """
    data = request.get_json()
    name = data.get("name")
    location = data.get("location")
    original_image_b64 = data.get("original_image_b64")

    if not all([name, location, original_image_b64]):
        return jsonify({
            "error": "Missing data (name, location, original_image_b64 required)."
        }), 400

    try:
        image_bytes = base64.b64decode(original_image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        top, right, bottom, left = location
        cropped_face = image.crop((left, top, right, bottom)).convert("RGB")

        # ---- Save face image locally -------------------------------------
        person_folder_name = name.replace(" ", "_").lower()
        person_path = os.path.join(KNOWN_FACES_PATH, person_folder_name)
        os.makedirs(person_path, exist_ok=True)
        file_count = len(os.listdir(person_path)) + 1
        face_path = os.path.join(person_path, f"face_{file_count}.jpg")
        cropped_face.save(face_path, "JPEG")

        # ---- Send to Dataiku for embedding enrollment --------------------
        buf = io.BytesIO()
        cropped_face.save(buf, format="JPEG")
        face_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            call_dataiku("/enroll_face", {"name": name, "face_b64": face_b64})
        except Exception as e:
            app.logger.error(
                f"Warning: Saved face locally but Dataiku enrollment failed: {e}",
                exc_info=True,
            )
            # Face is saved locally — Dataiku can be synced later via migration script

        return jsonify({"message": f"Saved '{name}' successfully."})

    except Exception as e:
        app.logger.error(f"Error in /add_face: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/recognize_faces", methods=["POST"])
def api_recognize_faces():
    """
    Recognize pre-cropped faces.  Forwards the request directly to Dataiku.
    """
    data = request.get_json()
    if not data or "faces" not in data or not isinstance(data["faces"], list):
        return jsonify({
            "error": 'Request must be JSON with a "faces" array.'
        }), 400

    try:
        result = call_dataiku("/recognize_faces", {"faces": data["faces"]})
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error in /recognize_faces: {e}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = os.environ.get("HTTP_PLATFORM_PORT", 5002)
    print(f"Starting Thumbnail API on port {port}...")
    print(f"  Dataiku Model API URL: {DATAIKU_API_URL}")
    print(f"  Known faces path:      {KNOWN_FACES_PATH}")
    serve(app, host="0.0.0.0", port=int(port), threads=100)
