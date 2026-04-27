@echo off
echo Starting Easy Transcriber...

if not exist .venv (
    echo [ERROR] Virtual environment not found. Please run 'setup.bat' first.
    pause
    exit /b 1
)

:: Activate venv and run app
call .venv\Scripts\activate
python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
