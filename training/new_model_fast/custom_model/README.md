# BirdNet v1 — custom lightweight detector (phase ②)

Single-class bird detector designed **from scratch** for the TI SK-AM62
(4×Cortex-A53, ONNX Runtime CPU/NEON). Target: same or better mAP50 than
production `best_448.onnx` (0.806 @448) at **≤ ~350 ms cycle** (prod:
~915 ms).

## Files

| file | what |
|---|---|
| `birdnet.py` | ReLU-only modules (`BConv`, `BC2f`, `BirdSPPF`, `BirdDetect`) + `register()` |
| `birdnet_v1.yaml` | the architecture (standard ultralytics yaml names, hardcoded channels) |
| `train_birdnet.py` | Colab VM runner: train from scratch @448 → validate → export ONNX → verify `(1,5,N)` |

## Design decisions

1. **ReLU everywhere.** SiLU = `x·sigmoid(x)` (mul+sigmoid+mul); ReLU is one
   `max`. On a ~90 GFLOPS NEON ceiling the elementwise layers matter.
   Channels kept at multiples of 8/16.
2. **Slim 1-class head.** Ultralytics' `Detect` sizes the cls branch from
   `max(ch[0], min(nc,100))` → 64 hidden channels even for `nc=1`. BirdNet
   uses 32/32 (box/cls hidden). DFL kept (16 bins) — accuracy is the honest
   risk, don't also handicap regression.
3. **Rides the ultralytics trainer.** The yaml uses the *standard* module
   names; `birdnet.register()` monkey-patches them in
   `ultralytics.nn.tasks` before the model is built. TAL assigner, DFL
   loss, mosaic, EMA, validator all inherited for free.
4. **Output contract matches `cpp/birdcpp.c` exactly.** Export is stock
   ultralytics → ONNX `(1, 4+nc, N)` = `(1, 5, N)`, cx/cy/w/h in letterbox
   pixels + sigmoided score; NMS stays in C. N = 4116 @448 (cap 8400).
5. **Trained at the deployment size (448)** — no train/deploy resolution
   mismatch, and cheap on the T4.

## Channel plan (v1)

backbone 16→32→64(P3/8)→96(P4/16)→128(P5/32)+SPPF; PAN neck out
64/96/128; cf. yolov8n: 16/32/64/128/256 backbone.

## Iteration protocol

Measure GFLOPs + params on the VM (`model.info()`), sanity-check mAP50 on
the 190-image val, export ONNX, then the real referee is the board: deploy
to the **sandbox** `/home/root/birdcpp-scratch/` on **:8890** (never touch
production :8888) and read the `/phone` stage timers.

Success gate: mAP50 ≥ 0.78 **and** cycle ≤ 350 ms.
