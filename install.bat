@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title J.A.R.V.I.S. Installer & Setup Wizard

echo ========================================================
echo       J.A.R.V.I.S. AI Assistant - Installation Setup
echo   Repository: https://github.com/shreyastouih-ctrl/project-jarvis
echo ========================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.11 or newer from https://www.python.org/downloads/
    echo (Make sure to check "Add Python to PATH" during installation)
    echo.
    pause
    exit /b 1
)

echo [1/4] Checking Python environment...
python --version

echo.
echo [2/4] Setting up Python virtual environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
    echo Virtual environment created.
) else (
    echo Existing virtual environment detected.
)

echo.
echo [3/4] Installing required dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
.venv\Scripts\python.exe -m pip install -e . --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation encountered an issue. Retrying with verbose output...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m pip install -e .
    if errorlevel 1 (
        echo [ERROR] Installation failed.
        pause
        exit /b 1
    )
)

echo.
echo [4/4] Creating Desktop Shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $desk = [Environment]::GetFolderPath('Desktop'); $s = $ws.CreateShortcut([System.IO.Path]::Combine($desk, 'JARVIS AI.lnk')); $s.TargetPath = '%~dp0start_jarvis.bat'; $s.WorkingDirectory = '%~dp0'; $s.Description = 'Launch J.A.R.V.I.S. AI Assistant'; $s.Save()" 2>nul
if exist "%USERPROFILE%\Desktop\JARVIS AI.lnk" (
    echo [OK] Shortcut created on your Desktop: "JARVIS AI"
) else (
    echo [INFO] Shortcut creation completed.
)

echo.
echo ========================================================
echo       SUCCESS: J.A.R.V.I.S. Is Installed and Ready!
echo ========================================================
echo - Desktop Shortcut : JARVIS AI.lnk
echo - Holographic HUD   : http://localhost:8000
echo - Auto-Updater     : Configured (checks every 90 days)
echo ========================================================
echo.

set /p launch="Would you like to launch J.A.R.V.I.S. now? (Y/N): "
if /i "%launch%"=="Y" (
    echo Launching JARVIS...
    start "" "%~dp0start_jarvis.bat"
)

echo.
echo Setup finished. Press any key to close.
pause >nul
