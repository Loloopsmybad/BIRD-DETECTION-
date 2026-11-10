CAMERA LAB (camkit) - standalone, OUTSIDE the bird system
=========================================================
Board folder : /home/root/camkit
PC mirror    : D:\samsung\BIRD-DETECTION\camkit\  (edit here, scp over, restart)
Web UI       : http://192.168.1.22:8889/
Service      : camkit.service   (ssh: systemctl start|stop|status camkit)

WHAT IT IS
  A live camera view plus every control the IMX219 driver exposes, so
  the sensor can be tuned by eye without touching the bird server:

    exposure (lines)          driver range shown on the page (per mode)
    analogue_gain             0..232 (~0..23.2 dB)
    digital_gain              256..4095 (256 = 1.0x, 512 = 2.0x)
    vertical_blanking         raw frame-time control (fps knob)
    frame-rate presets        30/25/20/15/10/5/2 fps -> computes vblank
                              from a self-measured line time
    test_pattern              0 off / 1 color bars / 2 solid / 3 grey / 4 PN9
    solid red + greenA + greenB + blue pixel values (patterns 2/3)
    horizontal / vflip        via driver flip controls (layout flags)
    sensor mode               1640x1232 (fastest) / 1920x1080 (default) /
                              3280x2464 (full sensor, slowest)
    view size                 half / 3-4 / full-frame (what the browser gets)
    view JPEG quality         30..95

  The fps meter on the page is measured from the raw stream - it is the
  ground truth. "~fps from vblank" estimates what the preset will give.
  Every Apply writes straight to the sensor; values re-load after each
  reconfiguration from the kernel itself.

SINGLE-CAMERA HANDOVER (no fights, by design)
  - Starting camkit stops birdboard automatically (camera is a
    single-lane resource; two servers cannot share /dev/video0).
  - Stopping camkit starts birdboard again automatically.
  - birdboard stays enabled at boot: on boot camkit pre-stops it, so
    whichever service you start last owns the camera.
  - Do NOT start birdboard by hand while camkit runs; if you did, stop
    birdboard again and camkit self-recovers within ~2 s.

HOW TO GO BACK TO BIRD DETECTION
  systemctl stop camkit        (birdboard comes back automatically)

HOW TO GO BACK TO THE CAMERA LAB
  systemctl start camkit       (birdboard is stopped automatically)

TECH NOTES (same proven recipe as the bird server)
  - capture: raw RG10 frames via plain os.read(); cv2.VideoCapture is
    deliberately NOT used (it mis-negotiates RG10 to 640x480 here).
  - bayer phase: re-verified at EVERY open using the sensor's solid-red
    test pattern (only the correct OpenCV CVT code yields R=255). This
    is why flips and mode switches can't silently break colors - the
    page shows the verified phase (BG / GB / RG / GR).
  - note: while the lab verifies the phase, the picture blinks red for
    ~0.5 s. That's the calibration, not a fault.
  - fps: the driver has no fps control; frame time = ((H + vblank) lines)
    x line_time. The line time is self-measured per mode/flip (18.9 us
    at 1080p, 1/29.99 s frame = 1080+683 lines).
  - exposure is in LINES: a long exposure also limits max fps. If you
    set 30 fps + huge exposure the sensor simply cannot keep up; lower
    exposure or accept the meter's lower number.
  - kernel chain: imx219 4-0010 -> cdns_csi2rx -> ticsi2rx -> video0.
  - The IMX219 device tree lives inside /boot/fitImage (see Bird Live
    System README for that story) - same caveat: protect the FIT).

ENDPOINTS (port 8889)
  /            lab UI                  /mjpg    live MJPEG stream
  /frame.jpg   latest frame JPEG       /status  JSON state (fps, values)
  /set         form apply (POST)       /readctl re-read values from kernel
  /ping        "camkit-ok"

DEPLOY FROM THE PC MIRROR
  scp "D:\samsung\BIRD-DETECTION\camkit\camkit.py" tibird:/home/root/camkit/
  ssh tibird "systemctl restart camkit"
  (enable once, already done: systemctl enable camkit)
