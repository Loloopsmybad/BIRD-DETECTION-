@echo off
cd /d D:\samsung\BIRD-DETECTION
venv\Scripts\python.exe camera_bridge.py "http://192.168.1.3:4747/video"
pause

