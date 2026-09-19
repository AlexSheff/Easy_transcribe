@echo off
title Easy Transcriber - Installation
cd /d "%~dp0"

echo ===================================================
echo   EASY TRANSCRIBER - INSTALLATION
echo ===================================================
echo.

node -v >nul 2>&1
if errorlevel 1 goto :no_node

echo [1/2] Node.js environment detected:
node -v
npm -v
echo.

echo [2/2] Installing dependencies via npm...
echo Please wait, this may take 1-2 minutes...
echo.

call npm install
if errorlevel 1 goto :npm_retry
goto :success

:npm_retry
echo.
echo [WARNING] Standard npm install failed.
echo Retrying with --legacy-peer-deps flag...
echo.
call npm install --legacy-peer-deps
if errorlevel 1 goto :install_error
goto :success

:no_node
echo.
echo ===================================================
echo [ERROR] Node.js is not installed or not in PATH!
echo ===================================================
echo.
echo Node.js is required to run Easy Transcriber locally.
echo.
echo 1. Download and install Node.js LTS from:
echo    https://nodejs.org/
echo 2. During installation, make sure "Add to PATH" is checked.
echo 3. After installing, close and reopen this window or restart PC.
echo.
echo ===================================================
pause
exit /b 1

:install_error
echo.
echo ===================================================
echo [ERROR] Failed to install dependencies.
echo ===================================================
echo Possible reasons:
echo - No internet connection
echo - Insufficient disk space or permissions
echo.
echo You can try running manually in Command Prompt:
echo   npm install
echo.
echo ===================================================
pause
exit /b 1

:success
echo.
echo ===================================================
echo [SUCCESS] All dependencies installed successfully!
echo You can now double-click 'run.bat' to start the app.
echo ===================================================
echo.
pause
exit /b 0
