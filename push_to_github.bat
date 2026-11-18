@echo off
set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%PATH%"
cd /d "%~dp0"
echo ============================================================
echo   Pushing project-jarvis to GitHub...
echo   Target: https://github.com/shreyastouih-ctrl/project-jarvis
echo ============================================================
git push -u origin main
echo.
pause
