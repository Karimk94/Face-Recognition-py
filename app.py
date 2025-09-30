from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
from face_processor import FaceProcessor
from waitress import serve
import os

# --- Initialization ---
app = Flask(__name__)
# Allow any origin for maximum flexibility
CORS(app, resources={r"/*": {"origins": "*"}})
face_processor = FaceProcessor()

@app.route('/api/analyze_image', methods=['POST'])
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
        # It's good practice to log the full exception for debugging
        app.logger.error(f"Error in /analyze_image: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/add_face', methods=['POST'])
def api_add_face():
    data = request.get_json()
    name = data.get('name')
    location = data.get('location')
    original_image_b64 = data.get('original_image_b64')
    if not all([name, location, original_image_b64]):
        return jsonify({'error': 'Missing data (name, location, original_image_b64 required).'}), 400
    try:
        image_bytes = base64.b64decode(original_image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        top, right, bottom, left = location
        cropped_face = image.crop((left, top, right, bottom))
        
        cropped_face = cropped_face.convert('RGB')
        
        buf = io.BytesIO()
        cropped_face.save(buf, format='JPEG')
        success = face_processor.add_new_face(name, buf.getvalue())
        if success:
            return jsonify({'message': f"Saved '{name}' successfully."})
        return jsonify({'error': 'Failed to save face.'}), 500
    except Exception as e:
        app.logger.error(f"Error in /add_face: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/recognize_faces', methods=['POST'])
def api_recognize_faces():
    data = request.get_json()
    if not data or 'faces' not in data or not isinstance(data['faces'], list):
        return jsonify({'error': 'Request must be JSON with a "faces" array.'}), 400
    
    base64_faces_list = data['faces']

    try:
        results = face_processor.recognize_faces_from_image(base64_faces_list)
        return jsonify({'faces': results})
    except Exception as e:
        app.logger.error(f"Error in /recognize_faces: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/analyze_image_stream', methods=['POST'])
def analyze_image_stream():
    """Endpoint to process a single image sent as a raw byte stream."""
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
        app.logger.error(f"Error in /analyze_image_stream: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    port = os.environ.get('HTTP_PLATFORM_PORT', 809)
    serve(app, host='localhost', port=port, threads=100)