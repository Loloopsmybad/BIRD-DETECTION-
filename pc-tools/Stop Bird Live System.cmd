@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*camera_bridge.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo Bridge stopped.
pause
