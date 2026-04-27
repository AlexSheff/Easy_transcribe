@echo off
SETLOCAL EnableDelayedExpansion

echo ======================================================
echo       Easy Transcriber - Windows Setup Script
echo ======================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

:: Create Virtual Environment
echo [1/3] Creating virtual environment...
if not exist .venv (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists. Skipping.
)

:: Install Dependencies
echo [2/3] Installing dependencies (this may take a few minutes)...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Check for FFmpeg
echo [3/3] Checking for FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] FFmpeg was not found in your system PATH.
    echo Easy Transcriber requires FFmpeg for audio extraction and processing.
    echo please download it from https://ffmpeg.org/ and add it to your PATH,
    echo or place ffmpeg.exe in the project root.
) else (
    echo FFmpeg detected.
)

echo.
echo ======================================================
echo Setup Complete!
echo You can now run the application using 'run.bat'
echo ======================================================
echo.
pause
