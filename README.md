# Bird Detection — live YOLOv8 on the TI SK-AM62 evaluation board

A deep-learning bird detection project (**YOLOv8-nano**, bounding boxes +
confidence, single class `bird`) with **two complete live operating modes**:

| mode | what runs where | status |
|---|---|---|
| **PC** | webcam / image / video detection scripts (`training\tools\`) | quick checks using the trained model |
| **Fully on the evaluation board** | the *entire* pipeline — camera capture, color processing, JPEG, detection inference with ONNX Runtime, annotation, and the HTTP dashboard — **runs locally on the TI SK-AM62 EVM board**, no PC, no server, no cloud | **production** |

This repo is written so that **everything needed to run live inference
locally on the evaluation board (TI SK-AM62 EVM)** is here: the C sources,
the python parity system, the camera-tuning lab, the shared capture stage,
the systemd units, and the full technical story of how we measured the
hardware, tuned the camera, and got every piece up and running.

---

## Live on the board right now

| thing | where |
|---|---|
| live dashboard (annotated MJPEG stream + settings) | <http://192.168.1.22:8888/> |
| clean (no boxes) stream | <http://192.168.1.22:8888/stream?raw=1> |
| detection meta JSON | <http://192.168.1.22:8888/meta.json> |
| camera status JSON | <http://192.168.1.22:8888/phone> |
| camera tuning lab (port 8889) | <http://192.168.1.22:8889/> |

Design in one paragraph: **live-media-only** — the snapshot JPEG endpoints
are gone; the camera runs *only* while someone watches `/stream`
(the dashboard's embedded `<img>` starts it). ~12 s after the last viewer
closes, the capture pump is SIGSTOP-ed and CPU drops to ~zero; the next
viewer resumes it instantly with the ORT session kept warm. Inference is
**fully native** — onnxruntime C API (dlopen), YOLOv8 NMS, box drawing and
the result JSON are all C; no python executes on the board while the C
service is up.

---

## What runs the inference on the board

| stage | detail |
|---|---|
| camera | Arducam V2.2 (IMX219) CSI, `/dev/video0`, RG10 Bayer, 4.15 MB/frame DMA |
| capture | `frame_pumper` — 4-thread C stage copying the uncached DMA buffer at ~21 fps through a pipe (see note below on the hardware wall) |
| color | 2x2 Bayer bin → gray-world AWB EMA → gamma 0.55 LUT |
| JPEG | libturbojpeg, declared locally (board ships no header) |
| detection | YOLOv8-nano `best.onnx` (also produced by `training\`) through onnxruntime 1.31.1 C API, 4 intra threads, ~1.2 s/forward |
| serve | C99 HTTP server in `cpp\birdcpp.c` — dashboard, MJPEG `/stream`, `/meta.json`, `/phone`, `/camera-set` |

That ~21 fps capture ceiling is **not** software slowness: the camera's DMA
memory reads at ~12 ms/MB uncached on this SoC, and every frame is 4 MB —
the pipes/threads exist to work around exactly that wall. The full
measurement table and every number's provenance: **[CAMERA_PIPELINE.md](CAMERA_PIPELINE.md)**.

---

## Repository layout

```
BIRD-DETECTION\
├── README.md                  <- this file
├── CAMERA_PIPELINE.md          <- technical pipeline doc + bring-up timeline (start here)
│
├── cpp\                        <- C VERSION - the production board system
│   ├── birdcpp.c               <- single-file C99 server (capture+infer+HTTP, ~2,900 lines)
│   ├── Makefile                <- gcc -O2 ... on the board (ORT include/lib paths)
│   ├── ort_probe.c             <- frozen feasibility probe for dlopen-ORT
│   ├── deploy\birdcpp.service  <- systemd unit
│   ├── README.md               <- build/deploy/run, endpoints, benchmarks, troubleshooting
│   └── reference\              <- frozen python snapshot at port time + retired sidecar
│
├── python\                     <- PYTHON VERSION - original live system (parity + handback)
│   ├── board_camera_server.py  <- the all-python birdboard (deployed at /home/root/bird)
│   ├── board_detection.py      <- YOLOv8 inference module (C version checked against it)
│   ├── best.onnx               <- deployed model copy (identical to training\checkpoints\best.onnx)
│   ├── README.md / README.txt  <- board info, addresses, services, troubleshooting
│   ├── artifacts\              <- frozen detection snapshots (python era)
│   └── legacy\                 <- early drafts kept for history
│
├── camkit\                     <- CAMERA TUNING LAB (python, port 8889)
│   └── camkit.py (+ service)   <- live view + every driver control + phase verify/WB
│
├── frame_pumper\               <- SHARED capture stage (C, source of the on-board binary)
│
├── pc-tools\                   <- PC-side ops: SSH/deploy shortcuts, camera bridge
│
├── training\                   <- the model itself
│   ├── Bird_Detection_Colab.ipynb   <- YOLOv8-nano fine-tune notebook (Colab T4)
│   ├── data_detection\         <- YOLO dataset (760 train / 190 val, 1 class)
│   ├── checkpoints\            <- best.pt (+ exported best.onnx = what the board runs)
│   ├── tools\                  <- PC webcam/image/video detection scripts
│   └── README.md               <- training usage, dataset format, tips
│
├── media\                      <- demo/model-test videos and output snapshots
└── venv\                       <- shared local python environment (PC side)
```

---

## How we got everything up and running (the short version)

The long version — measurements, kernel ABI tables, tuning math, traps — is
[`CAMERA_PIPELINE.md`](CAMERA_PIPELINE.md) (§6 is a chronological
bring-up timeline). The shape of it:

1. **Measure the hardware first.** Every capture path on the board was
   benchmarked; the uncached-DMA wall (~12 ms/MB) decided the whole design,
   and the 4-thread C frame_pumper became the one capture stage shared by
   everything.
2. **Tune the camera with a lab, not by feel.** `camkit` exposed every
   driver control; test-pattern lattice scoring pinned the Bayer phase;
   gray-world AWB + gamma LUT fixed the green cast.
3. **Get an end-to-end system working in python** (capture → inference →
   live dashboard, no disk writes) — this defined the routes, defaults and
   semantics everything later had to match byte-for-byte (`/camera-set`
   semantics are verbatim).
4. **Port to C99 in one file** (`cpp\birdcpp.c`): same routes, same
   defaults, turboJPEG locally declared — then reworked the design:
   snapshot endpoints removed, capture gated on viewers, inference moved
   fully into native ORT via dlopen.
5. **Operate it with the small toolkit** in `pc-tools\` (SSH/deploy
   shortcuts) and the docs: `cpp\README.md` for the C service,
   `python\README.txt` for board addresses/credentials, and the
   troubleshooting history that documents every footgun we actually hit.

---

## Quick start

### On the board (production path)

```bash
systemctl start birdcpp        # C birdbirdboard on :8888 (stops python birdboard)
```

Build/deploy details (scp, make, flags): [`cpp\README.md`](cpp/README.md).

### Back to the python board

```bash
systemctl stop birdcpp
systemctl start birdboard      # no auto-restart exists by design
```

### PC-side model checks

```
venv\Scripts\python.exe training\tools\camera_inference.py   # webcam
venv\Scripts\python.exe training\tools\test_image.py         # single image
venv\Scripts\python.exe training\tools\test_video.py         # video
```

(`venv\` must stay at the repo root; activation scripts hardcode it.)

---

## Model / training

YOLOv8-nano, fine-tuned on a 950-image bird dataset (760/190 split,
Colab T4, ~30 min) — dataset, notebook, checkpoints and accuracy tips:
[`training\README.md`](training/README.md).

---

### Service pointer cheat-sheet

| task | command |
|---|---|
| C birdboard live | `systemctl start birdcpp` (stops birdboard) |
| back to python birdboard | `systemctl stop birdcpp` → `systemctl start birdboard` manually |
| camera lab | `systemctl start camkit` (:8889) |
| logs | `journalctl -u birdcpp -f` |
| what's on the board | `python\README.txt` (addresses, passwords, folder map) |
