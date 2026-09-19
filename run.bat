@echo off
chcp 65001 >nul
title Easy Transcriber - Neuromicon Node

:: Ensure execution from the directory where this script is located
cd /d "%~dp0"

echo ===================================================================
echo   NEUROMICON // EASY TRANSCRIBER - ЗАПУСК / LAUNCHER
echo ===================================================================
echo.

:: Add default Node.js Windows installation paths to current session PATH if not already present
:: NOTE: Keep on single lines without parentheses to prevent (x86) in PATH from breaking cmd parser!
if exist "%ProgramFiles%\nodejs\node.exe" set "PATH=%ProgramFiles%\nodejs;%PATH%"
if exist "%ProgramFiles(x86)%\nodejs\node.exe" set "PATH=%ProgramFiles(x86)%\nodejs;%PATH%"
if exist "%LocalAppData%\Programs\node\nodejs\node.exe" set "PATH=%LocalAppData%\Programs\node\nodejs;%PATH%"
if exist "%AppData%\npm" set "PATH=%PATH%;%AppData%\npm"

where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ===================================================================
    echo  [ОШИБКА / ERROR] Node.js не найден в системе!
    echo ===================================================================
    echo  Сначала запустите install.bat или установите Node.js с https://nodejs.org/
    echo ===================================================================
    echo.
    pause
    exit /b 1
)

if not exist "node_modules\" (
    echo [ИНФО] Папка node_modules не найдена. Запуск предварительной установки install.bat...
    call install.bat
    if %ERRORLEVEL% NEQ 0 (
        echo [ОШИБКА] Установка не была завершена.
        pause
        exit /b %ERRORLEVEL%
    )
)

echo [ИНФО] Запуск локального узла Easy Transcriber...
echo [ИНФО] Приложение откроется по адресу: http://localhost:3000
echo [ИНФО] Чтобы остановить сервер, нажмите Ctrl+C в этом окне.
echo.

:: Open default browser after launching dev server
start "" "http://localhost:3000"

:: Start Vite development server
call npm run dev

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ИНФО] Сервер остановлен с кодом %ERRORLEVEL%.
)

pause

