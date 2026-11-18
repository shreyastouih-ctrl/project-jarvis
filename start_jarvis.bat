@echo off
cd /d "%~dp0"
title J.A.R.V.I.S. AI Assistant
echo ========================================================
echo          Starting J.A.R.V.I.S. AI Assistant
echo ========================================================
echo [INFO] Initializing Background Audio & Holographic HUD...
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_jarvis.py
) else (
    python run_jarvis.py
)
if errorlevel 1 (
    echo.
    echo [ERROR] JARVIS stopped unexpectedly. Press any key to exit.
    pause >nul
)
