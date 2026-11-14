@echo off
title Neural Style Transfer - Web Server
cd /d "%~dp0"
echo ========================================================
echo   Starting Neural Style Transfer Web Application...
echo   Mode: Fast AdaIN Inference (~1-2 seconds)
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ and make sure it is added to your PATH.
    pause
    exit /b 1
)

start "" http://localhost:5000
python server.py

pause
