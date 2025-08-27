from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import os
import random

class FaceProcessor:
    """
    Handles face detection and recognition using a robust, explicitly offline process.
    This version separates detection and recognition for maximum stability.
    """
    def __init__(self, known_faces_path="known_faces_db"):
        self.known_faces_path = known_faces_path
        os.makedirs(self.known_faces_path, exist_ok=True)
        self._verify_models_are_local()

    def _verify_models_are_local(self):
        """
        Checks if the necessary model files exist locally and are valid. If not,
        raises an exception to prevent the app from starting.
        """
        print("Verifying that AI models are available locally for offline use...")
        home = os.path.expanduser("~")
        weights_path = os.path.join(home, '.deepface', 'weights')
        
        vgg_face_model_path = os.path.join(weights_path, 'vgg_face_weights.h5')
        retinaface_model_path = os.path.join(weights_path, 'retinaface.h5')

        if not os.path.exists(vgg_face_model_path) or not os.path.exists(retinaface_model_path):
            error_message = (
                "CRITICAL ERROR: One or more model files are missing. "
                "Please run the 'setup.bat' script with an active internet connection."
            )
            raise RuntimeError(error_message)
        
        try:
            print("Attempting to load VGG-Face model to verify integrity...")
            DeepFace.build_model("VGG-Face")
            print("VGG-Face model loaded successfully.")
        except Exception as e:
            # This will catch errors from a corrupt file
            error_message = (
                f"CRITICAL ERROR: The model file 'vgg_face_weights.h5' is corrupt. Error: {e}\n"
                "Please follow these steps:\n"
                f"1. Go to this folder: {weights_path}\n"
                "2. Delete the file 'vgg_face_weights.h5'.\n"
                "3. Run the 'setup.bat' script again to re-download the file."
            )
            raise RuntimeError(error_message)

        print("✅ All models are available locally. Offline mode is ready.")


    def process_image(self, image_bytes):
        """
        Detects all faces, then separately attempts to recognize them for maximum stability.
        """
        random_number = random.randint(1, 999)
        temp_img_path = f"temp_image_for_processing_{random_number}.jpg"
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.save(temp_img_path, "JPEG")

            # --- Step 1: Detect ALL faces in the image ---
            print("Step 1: Detecting all faces in the image...")
            all_detected_faces = []
            try:
                all_detected_faces = DeepFace.extract_faces(
                    img_path=temp_img_path,
                    enforce_detection=False,
                    detector_backend='retinaface'
                )
            except Exception as e:
                print(f"🚨 'retinaface' backend failed: {e}. Trying fallback 'opencv' backend...")
                all_detected_faces = DeepFace.extract_faces(
                    img_path=temp_img_path,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
            print(f"Detection complete. Found {len(all_detected_faces)} faces.")

            # --- Step 2: Recognize known faces in the image (one single API call) ---
            recognized_faces = []
            db_is_populated = os.path.exists(self.known_faces_path) and len(os.listdir(self.known_faces_path)) > 0
            if db_is_populated:
                try:
                    print("Step 2: Recognizing known faces against the database...")
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
                    print(f"Recognition complete. Identified {len(recognized_faces)} known people.")
                except Exception as e:
                    print(f"Recognition step failed, will label all faces as Unknown. Error: {e}")

            # --- Step 3: Correlate detections with recognitions and draw results ---
            print("Step 3: Correlating results and drawing image...")
            image_for_drawing = Image.open(temp_img_path)
            draw = ImageDraw.Draw(image_for_drawing)
            results = {"faces": []}

            if not all_detected_faces or all_detected_faces[0]['facial_area']['w'] == 0:
                os.remove(temp_img_path)
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

                shadow_color = 'black'
                draw.text((x + 1, y - 21), label_text, font=font, fill=shadow_color)
                draw.text((x - 1, y - 19), label_text, font=font, fill=shadow_color)
                draw.text((x + 1, y - 19), label_text, font=font, fill=shadow_color)
                draw.text((x - 1, y - 21), label_text, font=font, fill=shadow_color)
                draw.text((x, y - 20), label_text, font=font, fill=(0, 255, 0))
                
                results["faces"].append({"index": face_number, "name": name, "location": [y, x + w, y + h, x], "distance": distance})

            buffer = io.BytesIO()
            image_for_drawing.save(buffer, format="JPEG")
            results["processed_image"] = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            os.remove(temp_img_path)
            return results

        except Exception as e:
            print(f"A critical error occurred during face processing: {e}")
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            raise e

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
            print(f"Error saving new face: {e}")
            return False