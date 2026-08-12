@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Complete the README setup first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\desktop_health_assistant.py --no-camera
if errorlevel 1 pause
