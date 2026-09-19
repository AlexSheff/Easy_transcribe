@echo off
title Easy Transcriber - Launcher
cd /d "%~dp0"

echo ===================================================
echo   EASY TRANSCRIBER - LAUNCHER
echo ===================================================
echo.

node -v >nul 2>&1
if errorlevel 1 goto :no_node

if not exist "node_modules\" goto :auto_install
goto :start_app

:auto_install
echo [INFO] node_modules folder not found.
echo Running install.bat first...
echo.
call install.bat
if errorlevel 1 goto :install_failed

:start_app
echo [INFO] Starting local development server...
echo [INFO] The application will open at: http://localhost:3000
echo [INFO] Press Ctrl+C in this window at any time to stop the server.
echo.

start "" "http://localhost:3000"
call npm run dev
if errorlevel 1 goto :dev_error
exit /b 0

:no_node
echo.
echo ===================================================
echo [ERROR] Node.js is not installed or not in PATH!
echo ===================================================
echo Please install Node.js from https://nodejs.org/ first.
echo.
pause
exit /b 1

:install_failed
echo.
echo ===================================================
echo [ERROR] Dependencies installation was not completed.
echo ===================================================
echo Please run install.bat and check for errors.
echo.
pause
exit /b 1

:dev_error
echo.
echo ===================================================
echo [INFO] Server stopped.
echo ===================================================
echo.
pause
exit /b 1
