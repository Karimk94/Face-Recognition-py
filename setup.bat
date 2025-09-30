@echo off
echo.
echo =================================================================
echo =  Setting up Face Recognition Service (Offline Mode)...
echo =================================================================
echo.

echo [1/3] Changing directory to the script's location...
cd /d "%~dp0"
echo.

echo [2/3] Creating Python virtual environment ('venv')...
python -m venv venv
if %errorlevel% neq 0 (
echo ERROR: Failed to create virtual environment. Ensure Python is installed correctly.
pause
exit /b
)
echo.

echo [3/3] Installing required Python libraries from local 'packages' folder...
if not exist packages (
echo ERROR: The 'packages' folder with library files was not found.
echo Please run 'pip download -r requirements.txt -d packages' on a machine with internet and include the folder.
pause
exit /b
)

REM --- The Key Command for Offline Installation ---
REM --no-index tells pip not to look online.
REM --find-links=./packages tells pip to look in our local folder for the library files.
venv\Scripts\python.exe -m pip install --no-index --find-links=./packages -r requirements.txt

if %errorlevel% neq 0 (
echo ERROR: Failed to install Python libraries from the local packages.
pause
exit /b
)
echo.

echo =======================================================
echo =  Setup Complete! The service is ready for IIS.
echo =======================================================
echo.
pause