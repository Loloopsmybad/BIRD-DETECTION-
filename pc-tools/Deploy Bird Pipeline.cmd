@echo off
REM Deploy Bird Pipeline.cmd - ship the PYTHON bird pipeline to the board
REM   (deploy set: ..\python\ -> /home/root/bird on tibird / 192.168.1.22)
REM For the C version (birdcpp) see ..\cpp\README.md - it has its own
REM build+maintained deploy steps on the board.
set SOPT=-o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa
set BOARD=tibird

echo == 1) pushing bird server files ==
scp %SOPT% "%~dp0..\python\board_camera_server.py" %BOARD%:/home/root/bird/board_camera_server.py || goto :fail
scp %SOPT% "%~dp0..\python\board_detection.py" %BOARD%:/home/root/bird/board_detection.py || goto :fail

echo == 2) pushing camkit (white-balance fix) ==
scp %SOPT% "%~dp0..\camkit\camkit.py" %BOARD%:/home/root/camkit/camkit.py || goto :fail

echo == 3) deploy + verify on-board ==
scp %SOPT% "%~dp0bird_deploy.sh" %BOARD%:/tmp/bird_deploy.sh || goto :fail
ssh %SOPT% %BOARD% "sh /tmp/bird_deploy.sh"
echo.
echo Look for: backend "pipe", fps ~14-16, and identical file counts
echo before/after (0 new images = capture removal verified).
pause
exit /b 0
:fail
echo DEPLOY FAILED - check the PC is on Saumya_5G and the board is on.
pause
