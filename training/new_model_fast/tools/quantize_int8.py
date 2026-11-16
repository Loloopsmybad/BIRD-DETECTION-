#!/usr/bin/env python
"""INT8 static quantization of a bird-detector ONNX for the AM62 board.

Phase (4) of new_model_fast: the AM62 forward is memory-bound (see PROGRESS
2026-11-16), so INT8 attacks both MACs and feature-map traffic. This tool
does onnxruntime static post-training quantization with real dataset images
as calibration, then smoke-verifies the quantized graph still decodes to the
birdcpp contract (1, 5, 4116) @448.

Typical use (PC, repo venv):
    venv\\Scripts\\python.exe training\\new_model_fast\\tools\\quantize_int8.py
        --onnx  training\\new_model_fast\\checkpoints\\birdnet_v1\\best_448.onnx
        --out   training\\new_model_fast\\checkpoints\\birdnet_v1\\best_448_int8.onnx
        --images "training\\V8 fine tuning\\data_detection\\train\\images" --n 128

Accuracy check afterwards:
    venv\\Scripts\\yolo.exe val model=<int8.onnx> imgsz=448 \\
        data=training\\new_model_fast\\data_bird.yaml
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required (repo venv has it via ultralytics)")

from onnxruntime.quantization import (  # noqa: E402
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

REPO = Path(__file__).resolve().parents[3]


def letterbox(im: np.ndarray, new_shape: int, color=(114, 114, 114)) -> np.ndarray:
    """Ultralytics-style letterbox to a square (BGR, no scaling to 32-pad extras)."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    if (nw, nh) != (w, h):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    bottom, right = new_shape - nh - top, new_shape - nw - left
    return cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


class LetterboxReader(CalibrationDataReader):
    """Feeds real dataset images, preprocessed exactly like birdcpp does:
    letterbox to imgsz, BGR->RGB, /255, HWC->CHW, float32."""

    def __init__(self, paths, input_name: str, imgsz: int = 448):
        self.paths = [p for p in paths]
        self.input_name = input_name
        self.imgsz = imgsz
        self.idx = 0

    def _blob(self, path):
        im = cv2.imread(str(path))
        if im is None:
            return None
        im = letterbox(im, self.imgsz)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = im.astype(np.float32) / 255.0
        return {self.input_name: x.transpose(2, 0, 1)[None]}

    def get_next(self):
        while self.idx < len(self.paths):
            d = self._blob(self.paths[self.idx])
            self.idx += 1
            if d is not None:
                return d
        return None

    def rewind(self):
        self.idx = 0


def list_images(folder: Path, n: int, seed: int = 0):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts]
    if not files:
        sys.exit(f"no images under {folder}")
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def find_last_conv_name(model):
    """Convs whose output reaches the output Sigmoid through pure shape /
    pointwise ops (Concat, Reshape, ...). Keeping those head convs fp32
    usually costs ~0 speed and recovers the most AP after PTQ."""
    nodes = list(model.graph.node)
    prod = {o: n for n in nodes for o in n.output}
    safe = {"Concat", "Reshape", "Mul", "Div", "Sub", "Add", "Sigmoid",
            "Transpose", "Slice"}
    out, seen = [], set()
    frontier = [i for s in nodes if s.op_type == "Sigmoid" for i in s.input]
    while frontier:
        t = frontier.pop()
        if t in seen:
            continue
        seen.add(t)
        n = prod.get(t)
        if n is None:
            continue
        if n.op_type == "Conv":
            out.append(n.name)
        elif n.op_type in safe:
            frontier.extend(n.input)
    return out


def smoke_compare(float_path: Path, int8_path: Path, paths, imgsz: int):
    """Run a few real images through both models; report drift + shape."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    f_sess = ort.InferenceSession(str(float_path), so, providers=["CPUExecutionProvider"])
    q_sess = ort.InferenceSession(str(int8_path), so, providers=["CPUExecutionProvider"])
    iname = f_sess.get_inputs()[0].name
    qiname = q_sess.get_inputs()[0].name
    reader = LetterboxReader(paths, iname, imgsz)
    reader.rewind()
    print(f"  float out: {f_sess.get_outputs()[0].shape}, int8 out: {q_sess.get_outputs()[0].shape}")
    ok = tuple(f_sess.get_outputs()[0].shape) == tuple(q_sess.get_outputs()[0].shape)
    diffs, sims = [], []
    for i, p in enumerate(paths[:6]):
        blob = reader._blob(p)
        if blob is None:
            continue
        yf = f_sess.run(None, {iname: blob[iname]})[0]
        yq = q_sess.run(None, {qiname: blob[qiname]})[0]
        a, b = yf.astype(np.float64).ravel(), yq.astype(np.float64).ravel()
        if a.shape != b.shape:
            print(f"  img {i}: SHAPE MISMATCH {yf.shape} vs {yq.shape}")
            ok = False
            continue
        mad = float(np.mean(np.abs(a - b)))
        denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
        cos = float(np.dot(a, b) / denom)
        diffs.append(mad)
        sims.append(cos)
        print(f"  img {i}: mean|diff|={mad:.4f}  cos={cos:.4f}")
    if diffs:
        print(f"  summary: mean|diff|={np.mean(diffs):.4f}  cos={np.mean(sims):.4f}")
    return ok


def cap_threads(n: int):
    """Patch ort.InferenceSession process-wide so calibration + smoke runs
    don't pin every core (desktop freeze guard, 2026-11-16). Handles both
    kwarg and positional sess_options callers."""
    import onnxruntime as ort

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
        so.intra_op_num_threads = n
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        k.pop("sess_options", None)
        k["sess_options"] = so
        return _orig(*pos, **k)

    ort.InferenceSession = _capped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True, help="float ONNX to quantize")
    ap.add_argument("--out", default=None, help="output int8 ONNX (default <stem>_int8.onnx)")
    ap.add_argument("--images", required=True, help="calibration image folder")
    ap.add_argument("--n", type=int, default=128, help="calibration sample count")
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--format", choices=["qop", "qdq"], default="qop",
                    help="QOperator (QLinearConv, best for ORT CPU EP) or QDQ")
    ap.add_argument("--acts", choices=["u8", "s8"], default="u8",
                    help="activation type (u8 classic mobile format)")
    ap.add_argument("--calib", choices=["minmax", "percentile", "entropy"], default="minmax")
    ap.add_argument("--percentile", type=float, default=99.9, help="for --calib percentile")
    ap.add_argument("--exclude-last", action="store_true",
                    help="keep the Conv feeding the output Sigmoid in fp32 "
                         "(usually recovers mAP at negligible speed cost)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0,
                    help="cap onnxruntime intra-op threads (0 = no cap). "
                         "Use e.g. 4 to keep the desktop responsive.")
    args = ap.parse_args()

    if args.threads:
        cap_threads(args.threads)

    onnx_path = Path(args.onnx)
    out_path = Path(args.out) if args.out else onnx_path.with_stem(onnx_path.stem + "_int8")
    files = list_images(Path(args.images), args.n, args.seed)
    print(f"calibrating on {len(files)} images from {args.images}")

    import onnx as _onnx

    model = _onnx.load(str(onnx_path))
    ops = sorted({n.op_type for n in model.graph.node})
    print("graph ops:", ops)
    op_types = [t for t in ("Conv", "MatMul") if t in ops]
    print("quantizing op types:", op_types)

    qf = QuantFormat.QOperator if args.format == "qop" else QuantFormat.QDQ
    at = QuantType.QUInt8 if args.acts == "u8" else QuantType.QInt8

    exclude = []
    if args.exclude_last:
        exclude = find_last_conv_name(model)
        if exclude:
            print("excluding fp32 node:", exclude)
        else:
            print("WARNING: no Conv feeding the output Sigmoid found")

    calib = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }[args.calib]

    reader = LetterboxReader(files, model.graph.input[0].name, args.imgsz)
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=qf,
        op_types_to_quantize=op_types,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=at,
        calibrate_method=calib,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "CalibTensorRangeSymmetric": False,
            "CalibMovingAverage": True,
            "CalibMaxIntermediateOutputs": False,
            "CalibPercentile": args.percentile,
            "NodesToExclude": exclude,
        },
    )
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB vs "
          f"{onnx_path.stat().st_size/1e6:.2f} MB float)")

    print("smoke compare (float vs int8):")
    ok = smoke_compare(onnx_path, out_path, files, args.imgsz)
    if not ok:
        sys.exit("shape mismatch between float and int8 outputs!")
    print("smoke OK")


if __name__ == "__main__":
    main()
