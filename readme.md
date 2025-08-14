Face Recognition Service API
This project is a standalone, offline-first Face Recognition API built with Flask. It is designed to be a robust, self-contained AI engine that can be called by any other application (such as an EDMS Viewer) to perform face analysis on images.
The service is built with a clear separation of concerns: it knows nothing about the applications that call it or the data sources they use. Its sole responsibility is to receive an image and return data about the faces within it.
🏛️ Architecture
Flask Backend: A lightweight and powerful web server that exposes the AI functionality through a simple REST API.
DeepFace Library: The core AI engine used for state-of-the-art face detection and recognition.
Offline-First Design: After an initial setup, the service runs 100% offline. It loads all AI models from the local machine and does not require an internet connection to operate.
CORS Enabled: The API is configured to accept requests from the EDMS Viewer application (running on http://127.0.0.1:5000), allowing for seamless communication between the two services.
Local Face Database: The service manages its own library of known individuals in a local known_faces_db folder. This makes the service portable and independent of any external database.
🚀 Getting Started
Follow these two steps to get the service up and running.
1. Initial Setup (One-Time Only)
This step installs all necessary libraries and downloads the required AI models for offline use. You will need an internet connection for this step.
Create Virtual Environment & Install Libraries: Double-click the setup.bat script. It will automatically create a Python virtual environment and install all the packages listed in requirements.txt.
Download AI Models: The setup script will then run download_models.py to download the face detection and recognition models (several hundred MBs) to your local user profile.
After this setup is complete, no internet connection is required.
2. Running the Service
To start the API server, simply double-click the run.bat script.
A command prompt window will open, and you will see messages indicating that the server is running on http://127.0.0.1:5001.
Leave this window open. The service is now active and ready to receive analysis requests from the EDMS Document Viewer or any other application. To stop the service, you can press Ctrl+C in this window or simply close it.
⚙️ API Endpoints
The service exposes the following endpoints:
POST /api/analyze_image
Receives an image file and returns a JSON object with the processed image and data for each detected face.
Request Body: multipart/form-data with a single field image_file.
Response:
{
  "processed_image_b64": "base64_string_of_image_with_boxes",
  "original_image_b64": "base64_string_of_original_image",
  "faces": [
    {
      "index": 1,
      "name": "john_doe",
      "location": [y1, x2, y2, x1]
    },
    {
      "index": 2,
      "name": "Unknown",
      "location": [y1, x2, y2, x1]
    }
  ]
}


POST /api/add_face
Receives a name and a cropped image of a face and saves it to the known_faces_db.
Request Body: application/json
{
  "name": "Jane Doe",
  "location": [y1, x2, y2, x1],
  "original_image_b64": "base64_string_of_the_full_image"
}


Response: A success or error message.
