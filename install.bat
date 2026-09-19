@echo off
setlocal
title Easy Transcriber - Installation

echo ===================================================
echo   NEUROMICON // EASY TRANSCRIBER - INSTALLATION
echo ===================================================
echo.

where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed or not found in system PATH.
    echo Please download and install Node.js (v18 or newer) from:
    echo   https://nodejs.org/
    echo.
    pause
    exit /b 1
)

echo [1/2] Node.js environment detected:
node --version
npm --version
echo.

echo [2/2] Installing application dependencies via npm...
call npm install
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Installation failed during npm install.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ===================================================
echo [SUCCESS] All dependencies installed successfully!
echo You can now double-click 'run.bat' to launch the app.
echo ===================================================
echo.
pause
