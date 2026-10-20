@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Starting the Face Recognition FastAPI server...
echo This service will run on http://127.0.0.1:5002
echo Leave this window open. Press Ctrl+C to stop the server.
echo.
python app.py
