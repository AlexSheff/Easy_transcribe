@echo off
chcp 65001 >nul
title Easy Transcriber - Installation

:: Ensure execution from the directory where this script is located
cd /d "%~dp0"

echo ===================================================================
echo   NEUROMICON // EASY TRANSCRIBER - УСТАНОВКА / INSTALLATION
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
    echo  [ОШИБКА / ERROR] Node.js не найден на вашем компьютере!
    echo ===================================================================
    echo.
    echo  Для запуска приложения требуется установленный Node.js (v18, v20 или новее LTS).
    echo.
    echo  Что нужно сделать:
    echo    1. Скачайте Node.js с официального сайта:
    echo       https://nodejs.org/
    echo    2. Установите его, обязательно оставив галочку "Add to PATH".
    echo    3. Если вы только что установили Node.js, перезапустите это окно
    echo       или перезагрузите проводник/ПК.
    echo.
    echo ===================================================================
    echo.
    pause
    exit /b 1
)

echo [1/2] Среда Node.js обнаружена:
call node --version
echo npm версия:
call npm --version
echo.

echo [2/2] Установка необходимых библиотек (npm install)...
echo Пожалуйста, подождите несколько секунд...
echo.

call npm install
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ПРЕДУПРЕЖДЕНИЕ] Стандартный 'npm install' завершился с кодом %ERRORLEVEL%.
    echo Пробуем установить с ключом --legacy-peer-deps...
    echo.
    call npm install --legacy-peer-deps
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ===================================================================
    echo  [ОШИБКА / ERROR] Не удалось установить зависимости через npm.
    echo ===================================================================
    echo  Возможные причины:
    echo   - Отсутствует подключение к интернету.
    echo   - Ограничения прав доступа к папке (попробуйте перенести папку
    echo     из системных папок на рабочий стол или диск D:).
    echo ===================================================================
    echo.
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo  [УСПЕХ / SUCCESS] Все библиотеки успешно установлены!
echo  Теперь вы можете запустить приложение, дважды кликнув на 'run.bat'.
echo ===================================================================
echo.
pause

