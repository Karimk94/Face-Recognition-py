@echo off
echo.
echo =======================================================
echo =  Setting up Face Recognition Service...
echo =======================================================
echo.

echo [1/3] Creating Python virtual environment ('venv')...
python -m venv venv
if %errorlevel% neq 0 (
echo ERROR: Failed to create virtual environment.
pause
exit /b
)
echo.

echo [2/3] Installing required Python libraries...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if %errorlevel% neq 0 (
echo ERROR: Failed to install Python libraries.
pause
exit /b
)
echo.

echo [3/3] Downloading AI models for offline use...
python download_models.py
if %errorlevel% neq 0 (
echo ERROR: Failed to download AI models.
pause
exit /b
)
echo.

echo =======================================================
echo =  Setup Complete!
echo =======================================================
echo You can now run the service using the 'run.bat' script.
echo.
pause