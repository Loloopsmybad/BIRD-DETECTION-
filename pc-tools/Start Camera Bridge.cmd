@echo off
REM Start Camera Bridge.cmd - legacy PC-bridge feeder
REM   (phone DroidCam -> this PC -> board; camera_bridge.py lives in this
REM    folder, repo venv is one level up)
REM Do NOT run while the board pulls the phone camera directly.
"%~dp0..\venv\Scripts\python.exe" "%~dp0camera_bridge.py" "http://192.168.1.3:4747/video"
pause
