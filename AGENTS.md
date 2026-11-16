# AGENTS.md — context for every AI session working in this repo

This file is auto-loaded into every OpenCode session in this directory.
Keep it short and current. Detailed history goes in `PROGRESS.md` (append-only).

## What this repo is

**Bird detection on an embedded board.** A camera feeds a TI SK-AM62 board
(`am62xx-evm`) which runs a single-file C program (`cpp/birdcpp.c`) doing ONNX
Runtime inference of a tiny object-detection model (YOLOv8-nano family, 1
class: `bird`), serving an MJPEG dashboard with boxes. Currently **production**
runs `best_448.onnx` @ 448 letterbox.

## Current project (started 2026-11-16)

`training/new_model_fast/` — **train a NEW model from scratch (not
fine-tuned)** with the goal: *lighter + faster on the board, same or better
accuracy than production (fwd ~875 ms / cycle ~915 ms / mAP50 0.806 @ 448)*.

Path agreed: (1) ultralytics `yolov8n.yaml` + `pretrained=False` baseline on
Colab T4 → (2) custom lightweight anchor-free detector in PyTorch, trained
from scratch → (3) distillation from the fine-tuned V8 → (4) INT8/ONNX for
the board. Log everything in the Obsidian vault (see below).

## Board (TI SK-AM62) — access & rules

- `ssh tibird` (configured in `~/.ssh/config`: root@192.168.1.22, key auth,
  ssh-rsa algorithm flags — dropbear is legacy; do not "fix" these flags).
- Board: 4×Cortex-A53 @1.4 GHz (~90 GFLOPS NEON ceiling), 2 GB RAM, Arago /
  kernel 5.10, python 3.8, **no pip**, gcc 9.2 (build on board).
- ONNX Runtime `1.31.1` (aarch64) at `/home/root/ort/`; birdcpp **dlopens**
  `libonnxruntime.so` — direct linking fails (no symbol-version records).
- systemd units (`/bin/systemctl`): `birdcpp` (production, :8888),
  `birdboard` (python fallback, :8888, disabled at boot), `camkit` (lab,
  :8889), `wifi-up`. Trial sandbox port: **8890**.
- **Benchmark protocol**: stage timers read 0 when idle (pipeline sleeps
  without viewers) — open `curl http://127.0.0.1:8888/stream` first, then
  read `/phone` JSON (stage timers snap/letterbox/fwd/nms/cycle).
- Gotchas: PowerShell mangles double-quoted remote commands (prefer
  single-quoted ssh args or run curls on the board); `pkill -f` self-matches
  its own shell → bracket trick `pkill -f 'bir[d]c[p]p'` + separate ssh
  calls; `killall -9 frame_pumper` fixes orphaned camera holds.
- **Trials protocol**: never touch production live. Sandbox = own folder
  (`/home/root/birdcpp-a/` style), port 8890, own journal prefix.

## Model lineage & measured numbers (val = 190 imgs / 413 birds)

| model | letterbox | fwd | cycle | mAP50 | mAP50-95 | recall |
|---|---|---|---|---|---|---|
| yolov8n best.onnx | 640 | ~1123-1735 ms | ~1810 ms | 0.832 | 0.662 | - |
| **best_448.onnx (PROD)** | 448 | ~875 ms | ~915 ms | 0.806 | 0.587 | 0.741 |
| best_384.onnx | 384 | ~717 ms | ~747 ms | 0.767 | - | 0.705 |
| YOLO-FastestV2 yf2_coco352 | 352 | ~120 ms | ~142 ms | **~0** | - | - |

- fv2 hit 7 Hz but ~0 AP on our birds (COCO checkpoint, wrong domain) → the
  parked conclusion was "**tiny model works only if retrained on our data**"
  — exactly what `new_model_fast` is doing.
- Board GPU (PowerVR AXE-1-16M) rejected for inference (no OpenCL/Vulkan
  compute, no ORT EP). No NPU/DSP on AM62 (that's AM62A). Remaining levers:
  smaller/quantized models. **INT8 quantization never attempted yet.**
- All three V8 ONNX sizes remain on the board in `/home/root/birdcpp/`;
  swap = edit `birdcpp.service --img` + `daemon-reload && restart`.

## Dataset & training assets

- `training/V8 fine tuning/data_detection/` — YOLO format, 760 train / 190
  val, 1 class `bird` (labels already fixed to class id 0; original Kaggle
  set used id 24). Source: kaggle `samuelayman/bird-dataset`.
- `training/V8 fine tuning/Bird_Detection_Colab.ipynb` — V8 fine-tune
  notebook (Colab T4, 50 epochs → mAP50 0.842 / mAP50-95 0.671).
- `training/V8 fine tuning/checkpoints/best.pt|best.onnx`; board exports
  via `training/V8 fine tuning/tools/export_trial_onnx.py <imgsz>` (local,
  no Colab needed).
- PC venv: repo-root `venv/` (hardcoded path, don't move). No NVIDIA GPU on
  this PC — all training on Colab T4 (GPU; TPUs don't work with
  ultralytics).
- Drive layout used by Colab notebooks: `MyDrive/BIRD/data_detection` (train
  input), results → `MyDrive/BIRD/<run-name>/`.

## Session coordination (for AI agents)

- **`PROGRESS.md`** — append-only log. After finishing a meaningful chunk of
  work, add a dated entry (one line per fact, latest at top). Read the top
  of it when starting a task.
- `tools/read_session.py` — read any other OpenCode session's history from
  the local DB: `venv\Scripts\python.exe tools\read_session.py` (list),
  `... read_session.py curious-mountain --last 20`. Useful: sessions
  `curious-mountain` (board deployment, Nov 4-10), `cosmic-lagoon` (Obsidian
  vault conventions).
- Obsidian notes vault:
  `C:\Users\Jay singh\OneDrive\Documents\Obsidian Vault\ARTIFICAIAL
  INTELLIGENCE\` — keep adding notes about what we do (user's standing
  request). `Project Logs/` folder holds project diaries; concept notes use
  full math + tables + `[[wiki-links]]`; maintain links in the
  `Artificial Intelligence.md` index.

## Repo layout (top level)

`cpp/` (birdcpp C + deploy), `python/` (legacy python board), `camkit/`
(camera lab), `frame_pumper/`, `pc-tools/` (PC ops kit), `media/`,
`training/` (V8 fine tuning | new_model_fast), `tools/` (cross-session
utilities), `venv/`.
