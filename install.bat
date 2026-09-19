@echo off
cd /d "%~dp0"
echo ===================================================
echo   EASY TRANSCRIBER - INSTALLING DEPENDENCIES
echo ===================================================
echo.
echo Current directory: %CD%
echo Running npm install, please wait...
echo.

call npm install --legacy-peer-deps

echo.
echo ===================================================
echo   INSTALLATION COMPLETED!
echo   You can now launch the app using run.bat or start.bat
echo ===================================================
echo.
pause
