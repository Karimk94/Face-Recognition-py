"""Local unit test: verify Dataiku envelope parsing (no network needed)."""
from unittest.mock import patch

from app.services import dataiku
from app.services.image import process_image_via_dataiku

# EXACT response captured from the staging server log (python_logs.log)
REAL_ENVELOPE = {
    "response": [
        {"bbox": {"x": 3646, "y": 1246, "w": 738, "h": 1017}, "confidence": 1.0},
        {"bbox": {"x": 1811, "y": 1173, "w": 723, "h": 1066}, "confidence": 1.0},
    ],
    "timing": {"preProcessing": 0, "wait": 62, "execution": 2476271},
    "apiContext": {"serviceId": "edms-fr-test", "endpointId": "detect_faces",
                   "serviceGeneration": "v5"},
}

# 1) detect_faces with the real envelope
with patch.object(dataiku, "_post", return_value=REAL_ENVELOPE):
    res = dataiku.detect_faces("fake_b64")
    assert len(res["faces"]) == 2, res
    assert res["faces"][0] == {"x": 3646, "y": 1246, "w": 738, "h": 1017,
                               "confidence": 1.0}
    print("1) detect_faces real-envelope      OK ->", res["faces"])

# 2) get_face_embeddings — list of plain vectors (with a None)
with patch.object(dataiku, "_post",
                  return_value={"response": [[0.1, 0.2], None], "apiContext": {}}):
    res = dataiku.get_face_embeddings(["a", "b"])
    assert res["embeddings"] == [[0.1, 0.2], None], res
    print("2) embeddings list-of-vectors      OK ->", res["embeddings"])

# 3) get_face_embeddings — list of {"embedding": [...]}
with patch.object(dataiku, "_post",
                  return_value={"response": [{"embedding": [1, 2, 3]}]}):
    res = dataiku.get_face_embeddings(["a"])
    assert res["embeddings"] == [[1, 2, 3]], res
    print("3) embeddings list-of-dicts        OK ->", res["embeddings"])

# 4) get_face_embeddings — legacy dict shape
with patch.object(dataiku, "_post", return_value={"embeddings": [[9]]}):
    res = dataiku.get_face_embeddings(["a"])
    assert res["embeddings"] == [[9]], res
    print("4) embeddings legacy dict          OK ->", res["embeddings"])

# 6) Batching: 12 images must be split into calls of 5, 5, 2 (order preserved)
calls = []
def fake_batch_post(endpoint, json_data):
    calls.append(len(json_data["images_b64"]))
    return {"response": [[float(i)] for i in range(len(json_data["images_b64"]))]}

with patch.object(dataiku, "_post", side_effect=fake_batch_post):
    res = dataiku.get_face_embeddings(["img"] * 12)
assert calls == [5, 5, 2], calls
assert len(res["embeddings"]) == 12, len(res["embeddings"])
print("6) batching 12 imgs -> 5+5+2 calls OK ->", calls)

# 7) Edge case: empty list -> no HTTP call at all
with patch.object(dataiku, "_post",
                  side_effect=AssertionError("should not be called")):
    res = dataiku.get_face_embeddings([])
assert res == {"embeddings": []}, res
print("7) empty list -> no HTTP call      OK")

# 5) Full pipeline end-to-end (mocked HTTP, real image + real ChromaDB)
with open("known_faces_db/ahmed_bahrozyan/face_1.jpg", "rb") as f:
    img_bytes = f.read()

VGG_FACE_DIM = 4096  # matches the dimension stored in chroma_db
def fake_post(endpoint, json_data):
    if "detect_faces" in endpoint:
        return REAL_ENVELOPE
    return {"response": [[0.01] * VGG_FACE_DIM, [0.02] * VGG_FACE_DIM]}

with patch.object(dataiku, "_post", side_effect=fake_post):
    # image.py imported detect_faces/get_face_embeddings by value, so patch there too
    import app.services.image as image_mod
    with patch.object(image_mod, "detect_faces", dataiku.detect_faces), \
         patch.object(image_mod, "get_face_embeddings", dataiku.get_face_embeddings):
        out = process_image_via_dataiku(img_bytes)

assert len(out["faces"]) == 2, out["faces"]
for face in out["faces"]:
    print("   pipeline face:", {k: v for k, v in face.items() if k != "thumbnail_b64"})
assert "processed_image" in out and "original_image_b64" in out
print("5) full pipeline (2 faces drawn)   OK")

print("\nALL PARSING TESTS PASSED")
