@echo off
setlocal

REM =================================================================
REM  Automated Packaging Script for Face Recognition API
REM  - Cleans Python cache
REM  - Creates a multi-part 7-Zip archive for easy transfer
REM =================================================================

echo.
echo === Face Recognition API Packaging Script ===
echo.

REM --- Step 1: Clean Python Cache ---
echo [1/4] Cleaning Python cache folders...
call clear_cache.bat
echo Cache cleared.
echo.

REM --- Step 2: Check for 7-Zip ---
echo [2/4] Checking for 7-Zip...
set "sevenzip_path=%ProgramFiles%\7-Zip\7z.exe"
if not exist "%sevenzip_path%" (
echo ERROR: 7-Zip is not found at "%sevenzip_path%".
echo Please install 7-Zip to use this packaging script.
echo You can download it from www.7-zip.org
pause
exit /b
)
echo 7-Zip found.
echo.

REM --- Step 3: Get Split Size from User ---
echo [3/4] Enter the size for each archive part in Megabytes (MB).
set /p split_size="Enter split size in MB (e.g., 1024 for 1GB): "
if not defined split_size set "split_size=1024"
echo.

REM --- Step 4: Create the Archive ---
echo [4/4] Creating multi-part archive. This may take a while...
set "output_dir=dist"
if not exist "%output_dir%" mkdir "%output_dir%"
set "archive_name=%output_dir%\deployment.7z"

REM Change directory to the script's location to package its contents
cd /d "%~dp0"

"%sevenzip_path%" a -t7z "%archive_name%" -v%split_size%m -r * -xr!"%output_dir%" -xr!package.bat -xr!clear_cache.bat

if %errorlevel% neq 0 (
echo.
echo ERROR: 7-Zip failed to create the archive.
pause
exit /b
)

echo.
echo ===============================================================
echo  Packaging Complete!
echo ===============================================================
echo Your deployment files are located in the '%output_dir%' folder.
echo Copy ALL of the '.7z.00X' files to your server.
echo.
pause