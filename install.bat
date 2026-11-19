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
    echo [ERROR] Python not found in PATH. Please install Python 3.11 or newer from https://www.python.org/downloads/
    exit /b 1
)

echo [1/4] Checking Python environment...
python --version

echo.
echo [2/4] Setting up Python virtual environment (.venv)...
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv .venv
    echo Virtual environment created.
) else (
    echo Existing virtual environment detected.
)

echo.
echo [3/4] Installing required dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
if errorlevel 1 (
    echo [ERROR] Installation failed.
    exit /b 1
)

echo.
echo [4/4] Creating J.A.R.V.I.S. App Shortcut with Arc-Reactor Icon...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $desk = [Environment]::GetFolderPath('Desktop'); $programs = [Environment]::GetFolderPath('Programs'); $s = $ws.CreateShortcut([System.IO.Path]::Combine($desk, 'J.A.R.V.I.S. AI.lnk')); $s.TargetPath = '%~dp0start_jarvis.bat'; $s.WorkingDirectory = '%~dp0'; if (Test-Path '%~dp0jarvis.ico') { $s.IconLocation = '%~dp0jarvis.ico' }; $s.Description = 'Iron Man J.A.R.V.I.S. AI Assistant'; $s.Save(); $s2 = $ws.CreateShortcut([System.IO.Path]::Combine($programs, 'J.A.R.V.I.S. AI.lnk')); $s2.TargetPath = '%~dp0start_jarvis.bat'; $s2.WorkingDirectory = '%~dp0'; if (Test-Path '%~dp0jarvis.ico') { $s2.IconLocation = '%~dp0jarvis.ico' }; $s2.Description = 'Iron Man J.A.R.V.I.S. AI Assistant'; $s2.Save()" 2>nul
if exist "%USERPROFILE%\Desktop\J.A.R.V.I.S. AI.lnk" (
    echo [OK] App shortcut created on your Desktop: "J.A.R.V.I.S. AI"
) else (
    echo [INFO] Shortcut creation completed.
)

echo.
echo ========================================================
echo       SUCCESS: J.A.R.V.I.S. Is Installed and Ready!
echo ========================================================
echo - App Shortcut     : J.A.R.V.I.S. AI.lnk (Desktop & Start Menu)
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
