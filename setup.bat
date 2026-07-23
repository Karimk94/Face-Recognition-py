@echo off
REM ============================================================================
REM  setup.bat — Setup script for the Face Recognition API
REM
REM  This script will:
REM  1. Delete any existing virtual environment (guarantees a clean state).
REM  2. Create a fresh Python virtual environment.
REM  3. Install all required packages OFFLINE from the local 'packages' folder.
REM  4. Create required runtime directories.
REM
REM  No internet connection is required — all wheels are in the 'packages' folder.
REM ============================================================================

@echo off
setlocal

echo Changing directory to the script's location...
cd /d "%~dp0"

REM ── Step 1: Remove existing venv ─────────────────────────────────────────────
echo.
echo [1/4] Removing existing virtual environment (if any)...
if exist venv (
    rmdir /s /q venv
    if %errorlevel% neq 0 (
        echo ERROR: Could not remove existing 'venv' folder.
        echo        Make sure no process is using it, then try again.
        pause
        exit /b 1
    )
    echo Removed old venv.
) else (
    echo No existing venv found. Skipping removal.
)

REM ── Step 2: Create fresh venv ────────────────────────────────────────────────
echo.
echo [2/4] Creating a fresh Python virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    echo        Please ensure Python 3.x is installed and on the PATH.
    pause
    exit /b 1
)
echo Virtual environment created.

REM ── Step 3: Install packages offline ─────────────────────────────────────────
echo.
echo [3/4] Installing dependencies OFFLINE from 'packages\' folder...
if not exist packages (
    echo ERROR: 'packages' folder not found.
    echo        Run 'download_packages.bat' on an internet-connected machine first,
    echo        then copy the 'packages' folder here.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
pip install --no-index --find-links=packages -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python packages from local 'packages' folder.
    echo        Ensure all required wheels are present. Re-run download_packages.bat
    echo        if needed.
    pause
    exit /b 1
)

REM ── Step 4: Create runtime directories ───────────────────────────────────────
echo.
echo [4/4] Creating runtime directories...
if not exist chroma_db  mkdir chroma_db
echo Directories ready.

echo.
echo =================================================================
echo  Setup Complete!
echo =================================================================
echo.
echo  Next steps for IIS deployment:
echo  1. In IIS Manager, create a new Site:
echo       Physical path : C:\edms\Face Recognition py
echo       Binding       : http, port 5002
echo  2. Grant the App Pool identity (IIS AppPool\<name>) Write access to:
echo       chroma_db\
echo.
echo  (Note: FastCGI registration is NO LONGER needed. IIS will automatically
echo   start the API using HttpPlatformHandler and uvicorn via web.config)
echo.
pause
endlocal