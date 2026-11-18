# JARVIS AI Assistant - PowerShell Installer
# Repository: https://github.com/shreyastouih-ctrl/project-jarvis.git
$ErrorActionPreference = 'Stop'
Write-Host '=== Installing JARVIS AI Assistant ===' -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "[ERROR] Python is not installed or not in PATH. Please install Python 3.11 or newer."
    exit 1
}
$VENV_DIR = Join-Path $PSScriptRoot '.venv'
if (-not (Test-Path $VENV_DIR)) {
    Write-Host 'Creating virtual environment...' -ForegroundColor Yellow
    python -m venv $VENV_DIR
}
$PYTHON_EXE = Join-Path $VENV_DIR 'Scripts/python.exe'
Write-Host 'Installing dependencies...' -ForegroundColor Yellow
& $PYTHON_EXE -m pip install --upgrade pip
& $PYTHON_EXE -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
& $PYTHON_EXE -m pip install -e $PSScriptRoot
Write-Host 'Installation completed successfully!' -ForegroundColor Green
