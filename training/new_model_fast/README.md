# new_model_fast — the from-scratch model project

Part of the **Bird Detection** repo. Goal: a model that is **lighter and
faster on the TI SK-AM62 board** than the current production `best_448.onnx`
(fwd ~875 ms / cycle ~915 ms) with **equal or better accuracy**
(mAP50 0.806 / mAP50-95 0.587 / recall 0.741 @ 448).

This is **training, not fine-tuning**: no pretrained weights anywhere —
every parameter starts random and learns from our 950-image bird dataset
only.

## Phases

| # | phase | status |
|---|-------|--------|
| 1 | Baseline: ultralytics `yolov8n.yaml` + `pretrained=False`, 250 epochs, Colab T4 | `bird.ipynb` ready to Run-all |
| 2 | Custom lightweight anchor-free detector, designed here, trained from scratch (PyTorch) | planned |
| 3 | Knowledge distillation: fine-tuned V8 teaches the custom model | planned |
| 4 | INT8 quantization + ONNX export → board trial on :8890 | planned |

Why from scratch at all? Because the board's ceiling is CPU (4×A53 NEON,
~90 GFLOPS — no usable GPU/NPU), so the only lever is a *smaller* model,
and the fastest small-model trick (YOLO-FastestV2 @ 7 Hz) scored ~0 mAP
with a COCO checkpoint — **tiny models only work here when retrained on
our data**. Training our own from scratch is both the experiment and the
requirement.

## Files

- `bird.ipynb` — Colab notebook (T4). Runtime → Run all → results in
  `MyDrive/BIRD/scratch_v1/`. Cell 8 is an OPTIONAL ssh/cloudflared bridge
  the PC session can request for interactive GPU work.

## Honest expectations for phase 1

From-scratch on 950 images will likely score **below** the fine-tuned V8
(0.842 mAP50) at first — the fine-tuned model inherited COCO features. What
we're measuring: how far from-scratch gets with the same architecture. The
custom architecture + distillation in phases 2-3 is where
"smaller AND accurate" comes from.

## Notes

Every step of this project is logged (with the why) in the Obsidian vault:
`...\Obsidian Vault\ARTIFICAIAL INTELLIGENCE\Project Logs\Bird Detector
From Scratch.md`, plus concept notes in the same vault.
Board context and measured numbers: repo-root `AGENTS.md`.
