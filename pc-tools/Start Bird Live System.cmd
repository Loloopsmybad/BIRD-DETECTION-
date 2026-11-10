@echo off
REM Start Bird Live System.cmd - LEGACY PC-bridge mode
REM   (DroidCam phone -> this PC's bridge on :9000 -> the board pulls from here)
REM Needs the repo venv. Do NOT run while the board pulls the phone camera
REM directly (DroidCam serves ONE client).
powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0Start-Live-System.ps1"
pause
