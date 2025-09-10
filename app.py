from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
from face_processor import FaceProcessor
from werkzeug.serving import run_simple

# --- Initialization ---
app = Flask(__name__)
# Allow any origin
CORS(app, resources={r"/*": {"origins": "*"}})
face_processor = FaceProcessor()

@app.route('/analyze_image', methods=['POST'])
def api_analyze_image():
    if 'image_file' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    file = request.files['image_file']
    image_bytes = file.read()
    try:
        results = face_processor.process_image(image_bytes)
        if results:
            results['original_image_b64'] = base64.b64encode(image_bytes).decode('utf-8')
            return jsonify(results)
        return jsonify({'error': 'Analysis returned no results.'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/add_face', methods=['POST'])
def api_add_face():
    data = request.get_json()
    name = data.get('name')
    location = data.get('location')
    original_image_b64 = data.get('original_image_b64')
    if not all([name, location, original_image_b64]):
        return jsonify({'error': 'Missing data.'}), 400
    image_bytes = base64.b64decode(original_image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    top, right, bottom, left = location
    cropped_face = image.crop((left, top, right, bottom))
    
    cropped_face = cropped_face.convert('RGB')
    
    buf = io.BytesIO()
    cropped_face.save(buf, format='JPEG')
    success = face_processor.add_new_face(name, buf.getvalue())
    if success:
        return jsonify({'message': f"Saved '{name}'."})
    return jsonify({'error': 'Failed to save face.'}), 500

@app.route('/recognize_faces', methods=['POST'])
def api_recognize_faces():
    data = request.get_json()
    if not data or 'faces' not in data or not isinstance(data['faces'], list):
        return jsonify({'error': 'Request must be JSON with a "faces" array.'}), 400
    
    base64_faces_list = data['faces']

    try:
        # Call the updated method in the face processor
        results = face_processor.recognize_faces_from_image(base64_faces_list)
        return jsonify({'faces': results})
    except Exception as e:
        print(f"ERROR in /recognize_faces: {e}")
        return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/analyze_image_stream', methods=['POST'])
def analyze_image_stream():
    """
    New endpoint to process a single image sent as a raw byte stream.
    """
    try:
        image_bytes = request.get_data()
        if not image_bytes:
            return jsonify(error="No image data in request body"), 400
        
        results = face_processor.process_image(image_bytes)
        
        if results:
            results['original_image_b64'] = base64.b64encode(image_bytes).decode('utf-8')
            return jsonify(results)
        return jsonify({'error': 'Analysis returned no results.'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error: {e}'}), 500

if __name__ == '__main__':
    run_simple(
        '127.0.0.1',
        5002,
        app,
        use_reloader=False,
        use_debugger=True,
        threaded=True,
        exclude_patterns=['*known_faces_db*', '*__pycache__*', '*venv*']
    )