@echo off
cd /d "%~dp0"
echo ===================================================
echo   EASY TRANSCRIBER - ALL-IN-ONE LAUNCHER
echo ===================================================
echo.
if not exist node_modules echo First-time setup: installing dependencies...
if not exist node_modules call npm install --legacy-peer-deps
echo.
echo Opening http://localhost:3000 in your browser...
start http://localhost:3000
call npm run dev
echo.
pause
