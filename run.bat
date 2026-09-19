@echo off
setlocal EnableDelayedExpansion
title Easy Transcriber - Neuromicon Node

:: Ensure execution from the directory where this script is located
cd /d "%~dp0"

echo ===================================================
echo   NEUROMICON // EASY TRANSCRIBER - LAUNCHER
echo ===================================================
echo.

:: Add default Node.js Windows installation paths to current session PATH if not already present
if exist "%ProgramFiles%\nodejs\node.exe" (
    set "PATH=%ProgramFiles%\nodejs;%PATH%"
)
if exist "%LocalAppData%\Programs\node\nodejs\node.exe" (
    set "PATH=%LocalAppData%\Programs\node\nodejs;%PATH%"
)

where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not found in your system PATH.
    echo Please run install.bat first or install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

if not exist "node_modules\" (
    echo [INFO] Dependencies not found. Running install.bat first...
    call install.bat
    if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
)

echo [INFO] Booting NEUROMICON Transcriber Node...
echo [INFO] Application will open at: http://localhost:3000
echo [INFO] Press Ctrl+C at any time in this window to stop.
echo.

:: Open default browser after a brief delay
start "" "http://localhost:3000"

:: Start Vite development server on port 3000
call npm run dev

pause
