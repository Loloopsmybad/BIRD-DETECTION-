BIRD LIVE SYSTEM / TI SK-AM62  (all-on-board edition)
=======================================================

THE PIPELINE (everything runs on the board)
--------------------------------------------
  Phone camera (DroidCam app, WiFi Server mode, MJPEG over your WiFi)
        |
        v  board reads the stream directly (no PC in the loop)
  TI SK-AM62 board:
    1. capture thread - parses MJPEG, auto-reconnects every 2 s
    2. YOLOv8 inference on the board (best.onnx, cv2.dnn)
    3. boxes drawn on the board -> live_det.jpg
    4. board hosts the viewer itself at:  http://192.168.1.22:8888/
  Open that URL on your PC or phone browser. That is the whole system.

BOARD LIFE CYCLE (automatic)
----------------------------
  Power on  ->  Wi-Fi connect -> birdboard.service starts -> camera re-grab
  Shut down ->  stream consumer vanishes -> phone camera stops being used
                (and the page/stream/inference simply stop with the board)
  Phone app must be open on the "WiFi Server" screen for reconnection.

ONLY ONE CAMERA CLIENT
----------------------
  DroidCam serves ONE client at a time. The board is the client.
  Do NOT also run the PC bridge while using the board directly, or the
  board loses the camera (it then shows "retrying every 2 s" and grabs
  the camera back as soon as the PC bridge is stopped).

FILES IN THIS FOLDER
--------------------
  View Inference Page.cmd      opens the dashboard (board-hosted)
  Get Latest Result.cmd        copies the latest annotated frame onto
                               Desktop\Bird Detection Results\live_det.jpg
  Start Bird Live System.cmd   LEGACY PC-bridge mode (needs PC python venv).
                               Only for the PC-as-middleman setup - do not
                               run it while the board holds the camera.
  board-files\
      board_camera_server.py   the program running on the board (birdboard.service)
      board_detection.py       YOLOv8 ONNX runner (cv2.dnn + numpy NMS)
      live_detect.sh           older PC-dependent loop (retired, kept for reference)
  camera_url.txt               (on the board: /home/root/bird/camera_url.txt)
                               the camera source the board pulls from.
                               Change the phone IP there if it ever changes.
  Model + venv: D:\samsung\BIRD-DETECTION\venv\ and checkpoints\best.onnx
                (the board has its own copy at /home/root/bird/best.onnx)

BOARD SERVICES (systemd)
-----------------------
  wifi-up.service    Wi-Fi connection AT BOOT - starts early (fixed ordering)
  birdboard.service  the whole live pipeline + viewer (main service; starts at boot)
  SSH                built-in dropbear of the board image (dropbird.service retired)
  Note: the page is normally reachable ~25 s after power-on. If it is not,
  give it up to ~1 min; the board retries the camera every 2 s on its own.
  (systemd-networkd-wait-online is masked: it stalled boot ~2 min waiting
  for the unplugged Ethernet port on this Wi-Fi-only board.)

EFFICIENCY (built in)
---------------------
  * Inference runs INSIDE the server process (no python restart per frame;
    the ONNX model is loaded ONCE and cached in the process)
  * A frame is detected once - identical frames are never re-run
  * JPEG bytes go decoder->memory directly (no temp-file write/read per frame)
  * No birds in the frame? The annotated file reuses the raw JPEG,
    no re-encode spent
  * Dashboard is feather-light: two image refreshes + tiny JSON polls,
    pauses all polling while the browser tab is hidden,
    and stops image fetches entirely while the camera is down
  * If no complete frame arrives for 15 s the board forces a reconnect
    (fixes phones that hold the socket but stop streaming)
  * /phone (status JSON) also reports a "frames" counter - climbing
    number = frames actually flowing from the camera

BOARD MEMORY (everything saved on the board itself)
---------------------------------------------------
  /home/root/bird/INFO.txt is the complete self-guide: addresses, endpoints,
  services, camera rules, how to change the camera source, troubleshooting.
  Read it from any machine: ssh tibird "cat /home/root/bird/INFO.txt"
  Board dashboard: http://192.168.1.22:8888/  (no PC required)

TROUBLESHOOTING
---------------
  Page down after boot?         wait ~1 min (Wi-Fi association is slowest)
  "camera: not connected"?      open DroidCam on the phone, tap WiFi Server
  Phone IP changed?             tell me, or edit /home/root/bird/camera_url.txt
  Board shell:                  ssh tibird   (root, password tiroot)
