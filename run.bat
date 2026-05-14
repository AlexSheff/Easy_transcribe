@echo off
echo Starting Easy Transcriber...

if not exist .venv (
    echo [WARNING] Virtual environment not found. Using global Python.
) else (
    if exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate.bat
    ) else if exist .venv\bin\activate.bat (
        call .venv\bin\activate.bat
    )
)

:: Run app
python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
