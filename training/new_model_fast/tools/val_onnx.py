#!/usr/bin/env python
"""Thread-capped + low-priority YOLO val wrapper for ONNX models on the PC.

onnxruntime defaults to one intra-op thread per logical core, so a plain
`yolo val model=x.onnx` pins every core at 100% for the whole run and can
freeze the desktop for ~30 s bursts (happened 2026-11-16). This wrapper
caps intra-op threads and sets below-normal process priority BEFORE
ultralytics builds its AutoBackend session.

Usage:
    venv\\Scripts\\python.exe training\\new_model_fast\\tools\\val_onnx.py \\
        --model <onnx> --data training\\new_model_fast\\data_bird.yaml
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--imgsz", type=int, default=448)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()

import onnxruntime as ort  # noqa: E402

_orig = ort.InferenceSession


def _capped(*a, **k):
    pos = list(a)
    so = None
    if len(pos) >= 2:  # positional sess_options slot: real options or None
        if isinstance(pos[1], ort.SessionOptions):
            so = pos[1]
        pos = [pos[0]] + pos[2:]
    if isinstance(k.get("sess_options"), ort.SessionOptions):
        so = k["sess_options"]
    if so is None:
        so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    k.pop("sess_options", None)
    k["sess_options"] = so
    return _orig(*pos, **k)


ort.InferenceSession = _capped

try:
    import psutil

    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

from ultralytics import YOLO  # noqa: E402

m = YOLO(args.model)
r = m.val(data=args.data, imgsz=args.imgsz, device="cpu")
res = {
    "model": os.path.basename(args.model),
    "P": float(r.box.mp),
    "R": float(r.box.mr),
    "mAP50": float(r.box.map50),
    "mAP50-95": float(r.box.map),
}
print("VAL_RESULT:", res)
