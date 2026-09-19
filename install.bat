@echo off
setlocal EnableDelayedExpansion
title Easy Transcriber - Installation

:: Ensure execution from the directory where this script is located
cd /d "%~dp0"

echo ===================================================
echo   NEUROMICON // EASY TRANSCRIBER - INSTALLATION
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
    echo [ERROR] Node.js is not installed or not found in system PATH.
    echo Please download and install Node.js (v18, v20 or newer LTS) from:
    echo   https://nodejs.org/
    echo.
    echo Note: When installing Node.js, ensure "Add to PATH" option is enabled.
    echo If you just installed Node.js, please restart your command prompt or computer.
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
    echo You can try running "npm install --legacy-peer-deps" manually.
    echo.
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
