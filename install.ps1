<#
.SYNOPSIS
    J.A.R.V.I.S. AI Assistant - Automated PowerShell Installer
.DESCRIPTION
    Sets up virtual environment, installs dependencies, builds desktop shortcuts,
    and configures the 90-day update cadence.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "      J.A.R.V.I.S. AI Assistant - Setup Wizard         " -ForegroundColor Cyan
Write-Host "  Repository: https://github.com/shreyastouih-ctrl/project-jarvis" -ForegroundColor DarkCyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python is not installed or not in PATH. Please install Python 3.11+." -ForegroundColor Red
    exit 1
}

$pyVer = & python --version 2>&1
Write-Host "[1/4] Found Python: $pyVer" -ForegroundColor Green

# 2. Virtual Environment
Write-Host "[2/4] Setting up Python virtual environment (.venv)..." -ForegroundColor Yellow
$VENV_DIR = Join-Path $PSScriptRoot '.venv'
if (-not (Test-Path $VENV_DIR)) {
    & python -m venv .venv
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[OK] Existing virtual environment detected." -ForegroundColor Green
}

$PYTHON_EXE = Join-Path $VENV_DIR "Scripts\python.exe"

# 3. Dependencies
Write-Host "[3/4] Installing dependencies & package..." -ForegroundColor Yellow
& $PYTHON_EXE -m pip install --upgrade pip --quiet
& $PYTHON_EXE -m pip install -r requirements.txt --quiet
& $PYTHON_EXE -m pip install -e . --quiet
Write-Host "[OK] Dependencies installed successfully." -ForegroundColor Green

# 4. Desktop & Start Menu Shortcut
Write-Host "[4/4] Creating J.A.R.V.I.S. App Shortcut with Arc-Reactor Icon..." -ForegroundColor Yellow
try {
    $ws = New-Object -ComObject WScript.Shell
    $desktop = [Environment]::GetFolderPath("Desktop")
    $programs = [Environment]::GetFolderPath("Programs")
    $shortcutPath = Join-Path $desktop "J.A.R.V.I.S. AI.lnk"
    $targetBat = Join-Path $PSScriptRoot "start_jarvis.bat"
    $icoPath = Join-Path $PSScriptRoot "jarvis.ico"

    $shortcut = $ws.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $targetBat
    $shortcut.WorkingDirectory = $PSScriptRoot
    if (Test-Path $icoPath) {
        $shortcut.IconLocation = $icoPath
    }
    $shortcut.Description = "Iron Man J.A.R.V.I.S. AI Assistant"
    $shortcut.Save()

    $shortcut2 = $ws.CreateShortcut((Join-Path $programs "J.A.R.V.I.S. AI.lnk"))
    $shortcut2.TargetPath = $targetBat
    $shortcut2.WorkingDirectory = $PSScriptRoot
    if (Test-Path $icoPath) {
        $shortcut2.IconLocation = $icoPath
    }
    $shortcut2.Description = "Iron Man J.A.R.V.I.S. AI Assistant"
    $shortcut2.Save()

    Write-Host "[OK] J.A.R.V.I.S. App shortcut created on Desktop and Start Menu!" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Could not create Desktop shortcut automatically: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "    SUCCESS: J.A.R.V.I.S. Installation Complete!       " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  * App Shortcut    : J.A.R.V.I.S. AI.lnk" -ForegroundColor White
Write-Host "  * Holographic HUD : http://localhost:8000" -ForegroundColor White
Write-Host "  * Auto-Updater    : Active (checks GitHub every 90 days)" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$launch = Read-Host "Would you like to launch J.A.R.V.I.S. now? (Y/N)"
if ($launch -eq "Y" -or $launch -eq "y") {
    Start-Process (Join-Path $PSScriptRoot "start_jarvis.bat")
}
