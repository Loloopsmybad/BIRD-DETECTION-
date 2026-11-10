@echo off
REM Get Latest Result.cmd - save ONE annotated detection snapshot and open it.
REM
REM   python birdboard (birdboard.service): /det.jpg -> works
REM   C birdcpp (birdcpp.service): /det.jpg is 410 Gone BY DESIGN -
REM     it is live-media-only (no disk writes). Watch /stream instead.
setlocal
set OUTDIR=D:\samsung\BIRD-DETECTION\media\pc-tool-outputs
set OUT=%OUTDIR%\last_det.jpg
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
curl -sf "http://192.168.1.22:8888/det.jpg" -o "%OUT%"
if errorlevel 1 goto :nostale
start "" "%OUT%"
exit /b 0
:nostale
echo.
echo Could not fetch /det.jpg from http://192.168.1.22:8888/det.jpg
echo   - C birdcpp running? Its snapshot endpoints are 410 Gone BY DESIGN.
echo     Watch the live dashboard: http://192.168.1.22:8888/
echo     (or swap to the snapshot-capable python board:
echo       ssh tibird "systemctl stop birdcpp; systemctl start birdboard")
echo   - Else: board off, or this PC not on the same wifi.
pause
exit /b 1
