@echo off
cd /d %USERPROFILE%\Desktop
if not exist "Bird Detection Results" mkdir "Bird Detection Results"
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'http://192.168.1.22:8888/live_det.jpg' -OutFile $\Env:USERPROFILE'\Desktop\Bird Detection Results\live_det.jpg'"
start "" "%USERPROFILE%\Desktop\Bird Detection Results\live_det.jpg"

