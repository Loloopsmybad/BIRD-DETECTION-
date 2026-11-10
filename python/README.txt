BIRD LIVE SYSTEM / TI SK-AM62
=============================
(Lives at D:\samsung\BIRD-DETECTION\Bird Live System — moved here from the
Desktop on Nov 6, 2026. The Desktop copy is retired.)


THIS FOLDER IS AN EXACT MIRROR OF THE BOARD'S /home/root/bird
-------------------------------------------------------------
Every file below (same names) is copied byte-for-byte from the board by scp
and verified by MD5. If you ever need to redeploy or rebuild the board,
this folder is the source of truth.

  board_camera_server.py   MAIN program on the board (birdboard.service):
                           capture thread + in-process YOLOv8 + web dashboard
  board_detection.py       YOLOv8 ONNX runner (ONNX net cached at load)
  best.onnx                the trained model
  INFO.txt                 the board's own self-guide (read it via ssh too)
  camera_url.txt           camera source ("/dev/video0" = CSI camera, live
                           now; "http://192.168.1.3:4747/video" = DroidCam)
  live_detect.sh           leftover reference script (not auto-run)
  diag_*.py / raw_diag.py  diagnostics used to debug camera/phone issues
  frame.jpg                latest raw camera frame (snapshot)
  live_det.jpg             latest annotated result (snapshot)
  meta.json                latest detections JSON (snapshot)
  24_*.jpg / live_*_det.jpg  test/debug frames captured during development
  __pycache__/             python bytecode cache (as on the board)

  _pc_tools\               RETIRED PC-era files kept only for reference
                           (camera_bridge.py, Start-Live-System.ps1, the old
                           .cmd launchers, SSH to TI Board.cmd, old README)
  Bird Detection Results\  where "Get Latest Result.cmd" copies annotations

BOARD INFORMATION (everything needed to operate without this PC)
----------------------------------------------------------------
  Board          : TI SK-AM62 (SK-AM62B-EVM), 4x Cortex-A53, 2 GB RAM
  Dashboard      : http://192.168.1.22:8888/   (opens from any device on Wi-Fi)
  Board IP       : 192.168.1.22  (Wi-Fi, SSID Saumya_5G)
  CSI camera     : Arducam IMX219 ("V2.2" cam) on the board's camera
                   connector -> /dev/video0 (details in the CSI section)
  Phone camera   : 192.168.1.3:4747 (DroidCam app -> WiFi Server screen ON)
  PC Wi-Fi IP    : 192.168.1.5 (only ever used to view/browse)

  SSH            : ssh tibird   (config in C:\Users\<user>\.ssh\config)
                   user root, key-based login (no password)
  Serial console : COM6 @ 115200  login root / password tiroot

  Services (systemd, enabled):
      birdboard.service  THE whole pipeline (camera + inference + dashboard)
      wifi-up.service    Wi-Fi association at boot (starts early)
      SSH                board image's built-in dropbear (dropbird retired)
      masked             systemd-networkd-wait-online (stalled boot ~2 min)

  Boot timing    : dashboard reachable ~ 25 s after power-on
  Self-healing   : camera retries every 2 s; no-frame stall -> reconnect;
                   stale ARP (phone MAC change) -> auto-flushed and retried
  Camera rules   : DroidCam serves ONE client - the board. Phone screen
                   unlocked keeps it streaming.

  Endpoints on port 8888:
      /            dashboard (light UI, self-hosted by the board)
      /frame.jpg   latest raw frame        /det.jpg    latest annotated
      /meta.json   detections JSON        /phone      camera link status
      /stream      full-screen MJPEG      /ping       health check

  Change the camera source (no PC needed):
      echo /dev/video0 > /home/root/bird/camera_url.txt        (CSI camera)
      echo http://NEWIP:4747/video > /home/root/bird/camera_url.txt (phone)
      re-read automatically; no restart needed (restart only reloads code)

CSI CAMERA (Arducam IMX219, Sony sensor) - LIVE SINCE NOV 6 2026
----------------------------------------------------------------
  Hardware: Raspberry-Pi-style Arducam V2.2 (IMX219) plugged into the
  SK-AM62 camera connector (15-pin ribbon).

  Kernel chain (measured 1080p30, 10-bit Bayer):
      imx219 4-0010 -> cdns_csi2rx -> ticsi2rx -> /dev/video0 .. 3

  Why this needed kernel work: the stock board image wires the OLD
  camera (OV5640) to the connector and has NO imx219 node, and the
  kernel/dtb pair is packed in a signed-looking u-boot FIT. Instead of
  fighting u-boot, the FIT was surgically rebuilt: stock fitImage byte
  layout kept 1:1 (same images, same configs, same names, same load
  addresses) with ONLY the data of the k3-am625-sk.dtb subimage
  replaced by [stock 1649-byte cert prefix + merged
  k3-am625-sk-imx219.dtb]. Verified by round-trip diff (exactly one
  property changed) and MD5 on both PC and board before/after the
  swap; the original is kept as /boot/fitImage-5.10.168-g2c23e6c538.bak
  (note /boot/fitImage is a symlink to the real file). Booted clean
  first try: "imx219 4-0010" probes, /dev/video0 appears.

  Capture path in board_camera_server.py (the CSI branch):
      plain blocking os.read() of raw RG10 frames (cv2.VideoCapture is
      NOT used: it mis-negotiates RG10 to 640x480 on this build)
      -> >>2 to 8-bit -> cv2.COLOR_BayerBG2BGR (phase PROVEN with the
      sensor's solid-red test pattern: only this code gives R=255)
      -> resize 1280x720 -> gray-world AWB, EMA-smoothed per channel,
      green reference, clamps 0.6-2.2 -> shadow-lift gamma LUT (0.55)
      -> JPEG q82 -> the same shared frame slot DroidCam used

  Tuning knobs (in board_camera_server.py, then scp + restart):
      CSI_GAIN        sensor analogue_gain, 0-232 (0-23.2 dB); set 210
      CSI_EXPOSURE    integration lines; 1759 = max for 1080p mode
      digital_gain    sensor ISP gain, written as 512 (2.0x) in
                      csi_configure()
      CSI_PREVIEW     dashboard/inference size, currently (1280, 720)

  On-board diagnostics (STOP birdboard first - the server owns
  /dev/video0 while running and busy-tests will fail with EBUSY):
      systemctl stop birdboard
      v4l2-ctl -d /dev/video0 --list-formats-ext
      v4l2-ctl -d /dev/v4l-subdev2 -l        (sensor controls: exposure,
                                              analogue_gain, test_pattern,
                                              digital_gain ...)
      media-ctl -d /dev/media0 -p            (pipeline topology)
      v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=5
  Sensor quirk: red/blue/green "pixel_value" controls only feed the
  sensor test-pattern generator (test_pattern=2 solid color); real
  brightness comes from exposure/analogue_gain/digital_gain only.

  Update code on the board (the scp round-trip):
      scp "<this folder>\board_camera_server.py" tibird:/home/root/bird/
      ssh tibird "systemctl restart birdboard"

EFFICIENCY (built in, all on board)
-----------------------------------
  * no subprocess per frame; ONNX model loaded once and cached
  * each frame detected once; JPEG decoded straight from memory
  * no-bird frames skip re-encoding (raw bytes reused)
  * dashboard is lightweight: two picture refreshes + tiny JSON polls,
    pauses while the browser tab is hidden or the camera is down
  * /stream throttled to 1 fps
  * /phone reports a "frames" counter: climbing = camera really flowing

PC-SIDE UTILITIES
-----------------
  _pc_tools\Get Latest Result.cmd   copies the newest annotated frame from
                                    the board to Desktop\Bird Detection
                                    Results\ and opens it (works as before)
  _pc_tools\SSH to TI Board.cmd     double-click shell into the board
  Everything else in _pc_tools is the retired PC-as-middleman pipeline.

QUICK CHECKS
------------
  Dashboard live?      open http://192.168.1.22:8888/
  Shell:               ssh tibird
  Board file mirror?   compare MD5: Get-FileHash <file> -Algorithm MD5
                       vs ssh tibird "md5sum /home/root/bird/<file>"
