@echo off
setlocal enabledelayedexpansion
title Easy TScribe - Neural Speech & Diarization Suite
cd /d "%~dp0"

echo ====================================================================
echo             EASY TSCRIBE - ALL-IN-ONE SYSTEM LAUNCHER
echo   Local Offline Whisper + Neural Speaker Diarization (100%% Private)
echo ====================================================================
echo.

:: 1. Verify Node.js and dependencies
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not found in system PATH.
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

if not exist node_modules (
    echo [*] First-time setup detected: installing frontend dependencies...
    call npm install --legacy-peer-deps
    if errorlevel 1 (
        echo [ERROR] npm install failed. Please verify your connection.
        pause
        exit /b 1
    )
)

:: 2. Ensure transcripts output folder exists in project directory
if not exist "transcripts" (
    mkdir "transcripts"
    echo [*] Initialized auto-save transcripts folder: %CD%\transcripts
)

:: 3. Launch local faster-whisper + diarization daemon if Python is available
where python >nul 2>&1
if not errorlevel 1 (
    echo [*] Python detected. Checking local faster-whisper daemon...
    echo [*] Strict offline mode active: HF_HUB_OFFLINE=1
    start "Easy TScribe - Local Python Engine" /min python server_faster_whisper.py
) else (
    echo [!] Python not detected. Running built-in WebGPU / browser transcription engine.
)

:: 4. Launch web application in browser and start dev server
echo.
echo [*] Starting Easy TScribe on http://localhost:3000 ...
echo [*] Generated markdown transcripts will automatically be saved to:
echo     %CD%\transcripts
echo.
timeout /t 2 >nul
start http://localhost:3000

call npm run dev

pause

