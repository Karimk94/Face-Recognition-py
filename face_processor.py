from deepface import DeepFace
from deepface.commons import folder_utils
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import os
import shutil
import time
import random

class FaceProcessor:
    """
    Handles face detection and recognition using a robust, explicitly offline process.
    This version ensures the deepface cache is populated before loading models.
    """
    def __init__(self, known_faces_path="known_faces_db"):
        self.known_faces_path = os.path.abspath(known_faces_path)
        os.makedirs(self.known_faces_path, exist_ok=True)
        self._initialize_offline_models()

    def _initialize_offline_models(self):
        """
        The definitive offline solution: Manually copies model files to the
        deepface cache location before telling deepface to build the model.
        This prevents any automatic download attempts.
        """
        print("Initializing offline models...")

        # This path is determined by the DEEPFACE_HOME environment variable set in web.config
        deepface_cache_path = folder_utils.get_deepface_home()
        weights_path = os.path.join(deepface_cache_path, 'weights')
        os.makedirs(weights_path, exist_ok=True)

        local_model_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
        
        required_models = ["vgg_face_weights.h5", "retinaface.h5"]

        for model_file in required_models:
            source_path = os.path.join(local_model_source_dir, model_file)
            dest_path = os.path.join(weights_path, model_file)

            if not os.path.exists(dest_path):
                print(f"Model '{model_file}' not found in cache. Copying from local source...")
                if not os.path.exists(source_path):
                    raise RuntimeError(f"CRITICAL ERROR: Source model file not found at '{source_path}'.")
                
                try:
                    # Pre-populate the deepface cache with our local model file
                    shutil.copy(source_path, dest_path)
                    print(f"Successfully copied '{model_file}' to cache.")
                except Exception as e:
                    raise RuntimeError(f"CRITICAL PERMISSION ERROR: Failed to copy model file from '{source_path}' to '{dest_path}'. Ensure the IIS user has 'Modify' permissions on the application folder. Error: {e}")

        try:
            # Now that the cache is guaranteed to be populated, build the model.
            # DeepFace will find the local files and will not attempt to download.
            print("Attempting to build VGG-Face model from pre-populated cache...")
            DeepFace.build_model("VGG-Face")

        except Exception as e:
            raise RuntimeError(f"CRITICAL ERROR: DeepFace failed to build the model even with a pre-populated cache. Error: {e}")

    def process_image(self, image_bytes):
        """
        Detects all faces, then separately attempts to recognize them for maximum stability.
        """
        # Using an absolute path for the temp image to avoid any ambiguity
        temp_img_path = os.path.join(os.path.dirname(__file__), f"temp_image_{random.randint(1,999)}.jpg")
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.save(temp_img_path, "JPEG")

            all_detected_faces = DeepFace.extract_faces(
                img_path=temp_img_path,
                enforce_detection=False,
                detector_backend='retinaface'
            )

            recognized_faces = []
            db_is_populated = os.path.exists(self.known_faces_path) and len(os.listdir(self.known_faces_path)) > 0
            if db_is_populated and all_detected_faces:
                try:
                    recognized_faces_dfs = DeepFace.find(
                        img_path=temp_img_path,
                        db_path=self.known_faces_path,
                        model_name="VGG-Face",
                        distance_metric="euclidean_l2",
                        enforce_detection=False,
                        detector_backend='retinaface',
                        silent=True
                    )
                    
                    for df in recognized_faces_dfs:
                        if not df.empty:
                            best_match = df.iloc[0]
                            identity_path = best_match['identity']
                            name = os.path.basename(os.path.dirname(str(identity_path)))
                            distance = best_match['distance']
                            recognized_faces.append({
                                'name': name,
                                'x': best_match['source_x'], 'y': best_match['source_y'],
                                'w': best_match['source_w'], 'h': best_match['source_h'],
                                'distance': distance
                            })
                except Exception as e:
                    print(f"Recognition step failed: {e}")

            image_for_drawing = Image.open(temp_img_path)
            draw = ImageDraw.Draw(image_for_drawing)
            results = {"faces": []}

            if not all_detected_faces or all_detected_faces[0]['facial_area']['w'] == 0:
                img_str = base64.b64encode(image_bytes).decode("utf-8")
                results["processed_image"] = img_str
                return results

            for i, detected_face in enumerate(all_detected_faces):
                facial_area = detected_face['facial_area']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                name = "Unknown"
                distance = None

                for rec_face in recognized_faces:
                    dist = np.sqrt(((x + w/2) - (rec_face['x'] + rec_face['w']/2))**2 + ((y + h/2) - (rec_face['y'] + rec_face['h']/2))**2)
                    if dist < 20 and rec_face['distance'] <= 1.0:
                        name = rec_face['name']
                        distance = rec_face['distance']
                        break

                draw.rectangle(((x, y), (x + w, y + h)), outline=(0, 255, 0), width=3)
                face_number = i + 1
                label_text = f"#{face_number}: {name.replace('_', ' ').title()}"
                
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()
                
                draw.text((x, y - 20), label_text, font=font, fill=(0, 255, 0))
                
                results["faces"].append({"index": face_number, "name": name, "location": [y, x + w, y + h, x], "distance": distance})

            buffer = io.BytesIO()
            image_for_drawing.save(buffer, format="JPEG")
            results["processed_image"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return results

        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    def add_new_face(self, name, cropped_face_bytes):
        """Saves a new person's face image to the known faces database directory."""
        try:
            person_folder_name = name.replace(" ", "_").lower()
            person_path = os.path.join(self.known_faces_path, person_folder_name)
            os.makedirs(person_path, exist_ok=True)
            
            file_count = len(os.listdir(person_path)) + 1
            face_path = os.path.join(person_path, f"face_{file_count}.jpg")
            
            with open(face_path, "wb") as f:
                f.write(cropped_face_bytes)
            
            pkl_file = os.path.join(self.known_faces_path, "representations_vgg_face.pkl")
            if os.path.exists(pkl_file):
                os.remove(pkl_file)

            return True
        except Exception as e:
            return False
        
    def recognize_faces_from_image(self, base64_faces_list):
        """
        Recognizes a list of pre-cropped faces provided as base64 strings.
        """
        final_results = []
        db_is_populated = os.path.exists(self.known_faces_path) and len(os.listdir(self.known_faces_path)) > 0

        for i, b64_face_string in enumerate(base64_faces_list):
            name = "Unknown"
            try:
                image_bytes = base64.b64decode(b64_face_string)
                img = Image.open(io.BytesIO(image_bytes))
                img_np = np.array(img)

                if db_is_populated:
                    dfs = DeepFace.find(
                        img_path=img_np,
                        db_path=self.known_faces_path,
                        model_name="VGG-Face",
                        distance_metric="euclidean_l2",
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if dfs and not dfs[0].empty:
                        best_match = dfs[0].iloc[0]
                        if best_match['distance'] <= 1.0:
                            identity_path = best_match['identity']
                            name = os.path.basename(os.path.dirname(str(identity_path)))
            except Exception as e:
                name = "Unknown"
            
            final_results.append({
                "index": i + 1,
                "name": name
            })
        
        return final_results