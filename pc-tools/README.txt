PC-SIDE TOOLKIT for the TI SK-AM62 bird system  (pc-tools)
==========================================================

This folder is the PC ops kit for the bird-detection board: SSH and deploy
shortcuts, the legacy PC camera bridge, and this guide. The board itself
runs every heavy part of the pipeline; nothing here does detection.

WHICH SYSTEM RUNS ON THE BOARD (192.168.1.22)
---------------------------------------------
  1) C version - PRODUCTION ("birdcpp.service", board folder /home/root/birdcpp)
       CSI camera (Arducam IMX219) + native ONNX Runtime inference in C,
       live-media-only: the snapshot endpoints are 410 Gone BY DESIGN,
       capture is gated on stream viewers (suspended ~12 s after the last
       one closes, ORT session kept warm). Sources: repo cpp\ with full
       docs in cpp\README.md.
       Dashboard  http://192.168.1.22:8888/   (annotated /stream default,
       /stream?raw=1 = clean) + /meta.json + /phone
  2) python version - parity/handback ("birdboard.service", /home/root/bird)
       the original all-python system, DroidCam phone camera over WiFi,
       snapshot capable: /det.jpg and /frame.jpg work. Sources: repo
       python\ ; board info in python\INFO.txt + python\README.txt.
  Only one of them can own the camera/port; starting birdcpp stops
  birdboard (no auto-restart of it afterwards - swap back manually).

QUICK COMMANDS
--------------
  Dashboard (C, production):   start http://192.168.1.22:8888/
  Board shell:                 ssh tibird            (root, password tiroot)
  C logs:                      journalctl -u birdcpp -f
  python board back:           ssh tibird "systemctl stop birdcpp; systemctl start birdboard"
  camera tuning lab (:8889):   ssh tibird "systemctl start camkit"
  Deploy PYTHON pipeline:      "Deploy Bird Pipeline.cmd" (python\ -> /home/root/bird)
  Deploy C pipeline:           see cpp\README.md (built on the board)

FILES IN THIS FOLDER
--------------------
  SSH to TI Board.cmd          board shell (ssh tibird)
  Deploy Bird Pipeline.cmd     scp python\ server + camkit\, then runs
                               bird_deploy.sh on the board (PYTHON era)
  bird_deploy.sh               board-side deploy+verify for the python
                               birdboard (its /det.jpg,/frame.jpg checks
                               answer 410 under the C birdcpp - by design)
  Get Latest Result.cmd        saves one /det.jpg snapshot to
                               media\pc-tool-outputs\last_det.jpg and
                               opens it. Works with the python board;
                               explains itself loudly on the C board.
  Start Bird Live System.cmd   LEGACY PC-bridge mode launcher (see below)
  Start-Live-System.ps1        finds the DroidCam phone, starts
                               camera_bridge.py (folder-local) with
                               bridge_config.txt as last-known-good URL,
                               opens http://localhost:9000/view
  Start Camera Bridge.cmd      bridge launcher variant (fixed phone URL)
  View Inference Page.cmd      opens the PC bridge viewer :9000/view
  Stop Bird Live System.cmd    kills any running PC camera_bridge.py
  camera_bridge.py             legacy PC middleman (python venv). Grabs
                               webcam or DroidCam, feeds the board on
                               :9000. LEGACY NOTE: it still mirrors the
                               result to
                               Desktop\Bird Detection Results\live_det.jpg
                               (DroidCam-one-PC era behavior, kept as-is)
  bridge_config.txt            last-good DroidCam URL for the bridge
                               (phone must run the DroidCam "WiFi Server"
                               screen; DroidCam serves ONE client - do not
                               run the bridge while the board pulls the
                               phone directly)

MODEL + ENV LOCATIONS
---------------------
  Trained weights (PC):        training\checkpoints\best.pt
  ONNX the board runs:         training\checkpoints\best.onnx
                               (identical copies ship in python\ and on
                               the board: /home/root/birdcpp/best.onnx,
                               /home/root/bird/best.onnx)
  PC python environment:       D:\samsung\BIRD-DETECTION\venv\
                               (must stay at the repo root - activation
                               scripts hardcode the path)

BOARD SERVICES (systemd, summary)
---------------------------------
  birdcpp.service     C production system (starts at boot; stops birdboard)
  birdboard.service   python birdboard (boot-kept too; needs manual start
                      after birdcpp stopped it)
  camkit.service      camera tuning lab on :8889
  wifi-up.service     Wi-Fi at boot (ordering fix; the page is normally
                      reachable ~25 s after power-on, give it ~1 min)
  SSH is the image's built-in dropbear.

TROUBLESHOOTING
---------------
  Page down after boot?         wait ~1 min (Wi-Fi association is slowest)
  "camera: not connected"?      open DroidCam, tap WiFi Server (python
                                board) - the C version uses the CSI camera
  Annotated stream blank?       fixed 2026-11-10 (canvas alloc bug); if it
                                ever returns, reload the dashboard tab and
                                check journalctl -u birdcpp for "bufs-ok"
  Capture stuck suspended?      open /stream in any tab - it resumes
                                within ~1 s (watch journalctl -u birdcpp
                                for "pumper resumed")
  Board shell:                  ssh tibird   (root, password tiroot)

Full technical story of the pipeline + hardware numbers: repo root
CAMERA_PIPELINE.md. C build/run/endpoints/benchmarks: cpp\README.md.
