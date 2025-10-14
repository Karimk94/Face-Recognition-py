#!/bin/bash

# --- Automated Setup and Deployment Script for Face Recognition App on macOS ---

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Face Recognition App Deployment ---"

# --- Configuration ---
# !!! IMPORTANT !!!
# UPDATE THIS VARIABLE to the absolute path where you have cloned the project.
PROJECT_DIR="Users/okool_arfelous/Developer/face-recognition-py"
# The name for the supervisor program
PROGRAM_NAME="face-recognition-py"
# The port for Gunicorn to listen on
PORT="8001"
# Your macOS username
MACOS_USERNAME=$(whoami)


# --- Step 1: Install Prerequisites with Homebrew ---
echo "--- Checking for Homebrew and installing prerequisites... ---"
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Please install it from https://brew.sh/ and re-run this script."
    exit 1
fi
brew install python supervisor


# --- Step 2: Set Up Project ---
echo "--- Setting up Python virtual environment... ---"
cd "$PROJECT_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate


# --- Step 3: Install Dependencies and Download Models ---
echo "--- Installing Python dependencies... ---"
pip install --upgrade pip
pip install -r requirements.txt

echo "--- Downloading machine learning models... ---"
python download_models.py
echo "--- Models downloaded successfully. ---"


# --- Step 4: Configure Supervisor ---
echo "--- Creating Supervisor configuration... ---"
SUPERVISOR_CONFIG_PATH="/usr/local/etc/supervisor.d/$PROGRAM_NAME.ini"
GUNICORN_PATH="$PROJECT_DIR/venv/bin/gunicorn"

# Create the supervisor config file
# Using printf to handle path variables safely
printf "[program:%s]\n" "$PROGRAM_NAME" | sudo tee "$SUPERVISOR_CONFIG_PATH" > /dev/null
printf "command=%s -w 4 -b 127.0.0.1:%s app:app\n" "$GUNICORN_PATH" "$PORT" | sudo tee -a "$SUPERVISOR_CONFIG_PATH" > /dev/null
printf "directory=%s\n" "$PROJECT_DIR" | sudo tee -a "$SUPERVISOR_CONFIG_PATH" > /dev/null
printf "user=%s\n" "$MACOS_USERNAME" | sudo tee -a "$SUPERVISOR_CONFIG_PATH" > /dev/null
printf "autostart=true\n" | sudo tee -a "$SUPERVISOR_CONFIG_PATH" > /dev/null
printf "autorestart=true\n" | sudo tee -a "$SUPERVISOR_CONFIG_PATH" > /dev/null
printf "stderr_logfile=/var/log/%s.err.log\n" "$PROGRAM_NAME" | sudo tee -a "$SUPERVISOR_CONFIG_PATH" > /dev/null
printf "stdout_logfile=/var/log/%s.out.log\n" "$PROGRAM_NAME" | sudo tee -a "$SUPERVISOR_CONFIG_PATH" > /dev/null

echo "--- Supervisor configuration created at $SUPERVISOR_CONFIG_PATH ---"


# --- Step 5: Start the Application with Supervisor ---
echo "--- Starting application with Supervisor... ---"
brew services restart supervisor
supervisorctl reread
supervisorctl update
supervisorctl start $PROGRAM_NAME

echo "--- Deployment complete! ---"
echo "Application '$PROGRAM_NAME' is now running."
echo "Check status with: supervisorctl status $PROGRAM_NAME"
echo "Access the application at: http://127.0.0.1:$PORT"
