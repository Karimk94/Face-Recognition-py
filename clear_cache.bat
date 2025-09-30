@echo off
REM =================================================================
REM  Utility Script to Clear Python Cache
REM  Run this before zipping your project for deployment.
REM =================================================================

echo Changing directory to the script's location...
cd /d "%~dp0"

echo Deleting all pycache folders...
for /d /r . %%d in (pycache) do @if exist "%%d" rd /s /q "%%d"

echo.
echo Python cache cleared successfully.
pause