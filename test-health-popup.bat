@echo off
rem Tests the layered, non-activating fullscreen health alert.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Complete the README setup first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\desktop_health_assistant.py --test-health-popup
if errorlevel 1 pause
