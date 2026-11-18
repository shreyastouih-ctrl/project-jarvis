@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
:: Repository: https://github.com/shreyastouih-ctrl/project-jarvis.git
echo === JARVIS AI Assistant Setup ===
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Please install Python 3.11 or newer.
    exit /b 1
)
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv .venv
)
echo [INFO] Installing required dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 (
    echo [ERROR] Installation failed.
    exit /b 1
)
echo [SUCCESS] JARVIS installed successfully.
