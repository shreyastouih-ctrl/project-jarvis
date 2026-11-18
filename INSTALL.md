# Installation Guide for JARVIS AI Assistant

**Official Repository**: [https://github.com/shreyastouih-ctrl/project-jarvis.git](https://github.com/shreyastouih-ctrl/project-jarvis.git)

## Prerequisites
- **Operating System**: Windows 10 / Windows 11 (64-bit)
- **Python**: Python >= 3.11 (tested on Python 3.13)
- Audio capture device (microphone) and output device (speakers or headphones)

## Installation

### Option A: One-Click PowerShell Script (Recommended)
Open PowerShell as Administrator or standard user and execute:
```powershell
.\install.ps1
```

### Option B: Windows Batch Launcher
Double-click `install.bat` or run from Command Prompt:
```cmd
install.bat
```

### Option C: Manual Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Running Jarvis
Launch the assistant using:
```powershell
python run_jarvis.py
# Or via installed script entry point:
jarvis
```

## Troubleshooting
- **Microphone Access**: Ensure microphone permissions are enabled in Windows Settings.
- **Audio Loopback / PortAudio**: If `sounddevice` fails to load, ensure PortAudio DLLs are in PATH.
- **Execution Policy**: If `install.ps1` is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
