# Python version - the python birdboard ("Bird Live System")

The **python version** of the live bird-detection system: the original
all-on-board python implementation (capture + color + JPEG + HTTP server +
inference), used as the parity reference for the C version and as the manual
handback option. Deployed on the TI SK-AM62 board at `/home/root/bird` as
`birdboard.service` (dashboard on **:8888**; the C version's service stops it
via `ExecStartPre` and there is no auto-restart anymore - start it manually
if you want the python board back).

## Files

| file | role |
|---|---|
| `board_camera_server.py` | the whole python birdboard (server, capture, inference host) |
| `board_detection.py` | YOLOv8 inference module (`run(...)` contract used by C-era code as parity reference) |
| `best.onnx` | the deployed ONNX model (identical to `..\training\checkpoints\best.onnx`) |
| `camera_url.txt` | camera source (= `/dev/video0`) |
| `diag_capture.py`, `diag_stream.py`, `raw_diag.py`, `live_detect.sh` | diagnostics + rolling-snapshot helper |
| `INFO.txt`, `README.txt` | board bring-up info (addresses, services, credentials, troubleshooting) |
| `legacy\` | early drafts (pre-fusion server, PC camera bridge v3) kept for history |
| `artifacts\` | frozen detection snapshots + rolling images from the python era |

> The current production path on the board is the **C version**: see
> `..\cpp\README.md`. The full bring-up story, hardware bottlenecks, camera
> tuning knowledge and both architectures' details are in
> `..\CAMERA_PIPELINE.md`.
