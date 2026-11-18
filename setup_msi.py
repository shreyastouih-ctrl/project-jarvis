import sys
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["jarvis", "sounddevice", "fastapi", "uvicorn", "websockets", "psutil", "edge_tts", "pydantic"],
    "include_files": [
        ("jarvis/ui/static", "jarvis/ui/static"),
        "README.md",
        "INSTALL.md",
    ],
    "excludes": ["pytest", "unittest"],
}

bdist_msi_options = {
    "upgrade_code": "{6A57E94E-2E31-4191-92D3-3D04C5A91A55}",
    "add_to_path": True,
    "initial_target_dir": r"[ProgramFilesFolder]\JARVIS AI",
    "summary_data": {
        "author": "Stark Industries",
        "comments": "J.A.R.V.I.S. AI Assistant Windows Installer",
    },
}

executables = [
    Executable(
        "run_jarvis.py",
        base=None,
        target_name="JARVIS.exe",
        shortcut_name="JARVIS AI Assistant",
        shortcut_dir="DesktopFolder",
    )
]

setup(
    name="JARVIS AI Assistant",
    version="1.0.0",
    description="Iron Man inspired J.A.R.V.I.S. AI Assistant",
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options,
    },
    executables=executables,
)
