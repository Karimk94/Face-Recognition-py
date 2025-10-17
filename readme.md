📸 Face Recognition Service API
A standalone, offline-first Face Recognition API built with Python and Flask. This service is designed as a robust, self-contained AI engine that can be called by any application to perform advanced face analysis on images.
🏛️ Core Architecture
🚀 Flask Backend: A lightweight and powerful web server that exposes the AI functionality through a simple REST API.
🧠 DeepFace Library: The core AI engine used for state-of-the-art face detection and recognition.
🔌 Offline-First Design: After the initial setup, the service runs 100% offline, loading all AI models from the local machine.
🌐 CORS Enabled: The API is configured to accept requests from any origin, allowing for seamless communication between services.
🗂️ Local Face Database: Manages its own library of known individuals in a local known_faces_db folder, making the service portable and independent.
🏁 Getting Started (Windows Development)
Follow these steps for local development and testing on a Windows machine.
1. Initial Setup (One-Time Only)
Note: An internet connection is required for this initial step.
Install Dependencies: Double-click the setup.bat script. It will automatically create a Python virtual environment and install all required packages.
Download AI Models: The setup script will then run download_models.py to download the necessary models into the project's models folder.
2. Running the Service
To start the API server, simply double-click the run.bat script.
A command prompt will open, and the server will be available at http://127.0.0.1:5001. Keep this window open to keep the service active.
🚀 Deployment on macOS (Production)
This guide details how to deploy the application as a persistent background service on macOS using Homebrew and Supervisor. This is the recommended method for a stable, production-like environment that automatically restarts on boot or on failure.
Step 1: Install Prerequisites
Install Python and Supervisor using Homebrew.
brew install python
brew install supervisor


Step 2: Start Supervisor as a Global Service
Configure supervisor to launch automatically at login. This ensures your application is always running.
brew services start supervisor


Step 3: Prepare the Application
Clone and Enter the Repository:
git clone [https://github.com/karimk94/face-recognition-py.git](https://github.com/karimk94/face-recognition-py.git)
cd face-recognition-py


Create and Activate Virtual Environment:
python3 -m venv venv
source venv/bin/activate


Install Python Dependencies:
pip install --upgrade pip
pip install -r requirements.txt


Download AI Models:
python download_models.py


Step 4: Configure Supervisor
Locate the Supervisor Configuration Directory. Use brew --prefix to find the correct path for your system:
Apple Silicon (M1/M2/M3): /opt/homebrew/etc/supervisor.d/
Intel: /usr/local/etc/supervisor.d/
Create the Directory if it doesn't already exist:
mkdir -p $(brew --prefix)/etc/supervisor.d/


Create a New Configuration File for the application:
touch $(brew --prefix)/etc/supervisor.d/face_recognition_api.conf


Add the Configuration. Open the file and paste the following, making sure to replace placeholder paths with the correct absolute path to your project folder.
[program:face-recognition-api]

; Command to run the waitress production server from the project's virtual environment.
command=/Users/your_user/Developer/face-recognition-py/venv/bin/waitress-serve --host 127.0.0.1 --port 5004 app:app

; The absolute path to the project directory.
directory=/Users/your_user/Developer/face-recognition-py/

autostart=true
autorestart=true

; Log file paths (ensure you have permissions to write to /var/log or choose another location).
stderr_logfile=/var/log/face_recognition_api.err.log
stdout_logfile=/var/log/face_recognition_api.out.log


Step 5: Launch the Application
Reload Supervisor's Configuration to detect your new file:
supervisorctl reload


Check the Status to confirm it's running:
supervisorctl status
You should see an output like: face-recognition-api RUNNING pid 12345, uptime 0:01:23
Your API is now running as a managed background service!
⚙️ Troubleshooting Deployment Issues
Here are solutions to common errors encountered during macOS deployment.
Problem 1: no config updates to processes
This message appears after running supervisorctl reread. It means Supervisor did not find your new .conf file.
Cause: The main supervisord daemon is running with an outdated configuration and hasn't registered the [include] directive correctly.
Solution: Use the reload command. This forces Supervisor to perform a full shutdown and restart, reading all configuration files from scratch.
supervisorctl reload


Problem 2: Another program is already listening on a port
This error occurs if you try to run supervisord -n for debugging while the global service is already running.
Cause: Two instances of supervisord are trying to use the same communication port.
Solution: Ensure only one instance is running. Shut down everything and start fresh with the global Homebrew service.
# Stop the global service if it's running
brew services stop supervisor

# Forcefully terminate any lingering supervisord processes
pkill supervisord

# Start fresh with the global service only
brew services start supervisor


Problem 3: WARN No file matches via include '*.ini'
This warning indicates a misconfiguration in the main supervisord.conf file.
Cause: The main config file is telling Supervisor to look for files ending in .ini, but your file ends in .conf.
Solution: Edit the main configuration file to look for the correct extension.
Open the file (e.g., /opt/homebrew/etc/supervisord.conf on Apple Silicon).
Find the [include] section and change the files line:
; Change this:
; files = /path/to/supervisor.d/*.ini

; To this:
[include]
files = /opt/homebrew/etc/supervisor.d/*.conf


Save the file and restart the service: brew services restart supervisor.
📡 API Endpoints
POST /api/analyze_image: Receives an image file, returning a JSON object with a processed image and data for each detected face.
POST /api/add_face: Receives a name and a cropped face image, saving it to the known faces database.
