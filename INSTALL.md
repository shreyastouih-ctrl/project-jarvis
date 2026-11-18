# Installation Guide for JARVIS AI Assistant

**Official Repository**: [https://github.com/shreyastouih-ctrl/project-jarvis.git](https://github.com/shreyastouih-ctrl/project-jarvis.git)

## Installation Options

### Option A: Windows MSI Installer (Recommended)
Double-click the official standalone Windows Installer:
- **`dist/jarvis-assistant-0.1.0-win64.msi`**
This provides a native Windows Setup Wizard that installs J.A.R.V.I.S. into `Program Files`, packages the runtime and dependencies, and creates a Start Menu & Desktop Shortcut automatically!

### Option B: Windows Batch Setup Wizard
Double-click `install.bat` or run:
```cmd
install.bat
```

### Option C: One-Click PowerShell Script
Open PowerShell and run:
```powershell
.\install.ps1
```

### Option D: Manual Developer Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Running Jarvis
Launch the assistant using the Desktop Shortcut created by the installer, or run:
```powershell
python run_jarvis.py
# Or:
start_jarvis.bat
```

## Troubleshooting
- **Microphone Access**: Ensure microphone permissions are enabled in Windows Settings.
- **Audio Loopback / PortAudio**: If `sounddevice` fails to load, ensure PortAudio DLLs are in PATH.
- **Execution Policy**: If `install.ps1` is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
