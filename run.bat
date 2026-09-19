@echo off
cd /d "%~dp0"
echo ===================================================
echo   EASY TRANSCRIBER - STARTING APPLICATION
echo ===================================================
echo.
echo Opening http://localhost:3000 in your browser...
echo.
start http://localhost:3000
call npm run dev
echo.
pause
