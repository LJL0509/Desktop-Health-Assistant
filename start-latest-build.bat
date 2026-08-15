@echo off
cd /d "%~dp0"
set "APP=dist\Desktop Health Assistant\DesktopHealthAssistant.exe"

if not exist "%APP%" (
    echo Packaged application not found. Run build-windows.ps1 first.
    pause
    exit /b 1
)

start "" "%APP%"
