#!/usr/bin/env python
"""distill_birdnet.py - phase (3): response distillation for BirdNet.

Teacher: fine-tuned YOLOv8n (COCO->birds, mAP50 0.806 @448). Student:
BirdNet v1 architecture (custom ReLU-only, 1.23M params) trained with the
STOCK ultralytics trainer - we only wrap the loss criterion.

Distillation design (version-robust, no internals surgery):
  - teacher (frozen, eval) and student (eval-mode second forward) both emit
    the decoded graph output (B, 4+nc, N) = (B, 5, 4116) @448: [xywh px,
    score]. Same anchor grid (strides 8/16/32) -> anchors align 1:1.
  - distill loss = teacher-score-weighted SmoothL1(student_xywh,
    teacher_xywh) + MSE(student_score, teacher_score), divided by the
    weight mass (proper weighted mean over anchors).
  - lambda ramps in over --ramp epochs (eval-mode BN stats are cold early).
  - supervised base loss (TAL/box/cls/DFL) stays exactly stock.

Runs on the Colab VM (driven over colab-mcp):

    python distill_birdnet.py --name birdnet_d1 > birdnet_distill.log 2>&1 &

Optional warm start from our own v1 artifact (still "not COCO fine-tuned"):
    --init /content/drive/MyDrive/BIRD/birdnet_v1/best.pt
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
import types

LOG = lambda *a: print(*a, flush=True)

DISTILL = {"epoch": -1, "calls": 0}  # mutable state, fed by callback


def find_dataset():
    local = "/content/data_detection"
    if os.path.isdir(os.path.join(local, "train", "images")):
        return local, False
    for c in (
        "/content/drive/MyDrive/BIRD/data_detection",
        "/content/drive/MyDrive/BIRD/bird_data_detection",
    ):
        if os.path.isdir(os.path.join(c, "train", "images")):
            return c, True
    for h in glob.glob("/content/drive/MyDrive/*/*/*/"):
        if os.path.basename(h.rstrip("/")) == "data_detection" and os.path.isdir(
            os.path.join(h, "train", "images")
        ):
            return h, True
    return None, False


def find_teacher(explicit):
    """Locate the fine-tuned V8 teacher .pt (never a birdnet run artifact)."""
    if explicit and os.path.exists(explicit):
        return explicit
    cands = []
    for pat in (
        "/content/teacher.pt",
        "/content/drive/MyDrive/BIRD/**/best.pt",
        "/content/drive/MyDrive/BIRD/*.pt",
        "/content/drive/MyDrive/**/best.pt",
    ):
        cands += [p for p in glob.glob(pat, recursive=True) if os.path.isfile(p)]
    cands = sorted(set(cands))
    # deprioritize our own birdnet artifacts
    birdnet = [c for c in cands if "birdnet" in c.lower() or "birdnet_runs" in c]
    rest = [c for c in cands if c not in birdnet]
    LOG("teacher candidates:", cands)
    return rest[0] if rest else None


def fix_labels(root):
    fixed = 0
    for lbl in glob.glob(os.path.join(root, "**", "*.txt"), recursive=True):
        with open(lbl) as f:
            lines = f.readlines()
        new_lines, changed = [], False
        for line in lines:
            parts = line.split()
            if parts and parts[0] != "0":
                parts[0] = "0"
                changed = True
            new_lines.append(" ".join(parts) + "\n")
        if changed:
            with open(lbl, "w") as f:
                f.writelines(new_lines)
            fixed += 1
    return fixed


def decoded_output(raw):
    """(B, 4+nc, N) from a model's eval-mode return value."""
    y = raw[0] if isinstance(raw, (tuple, list)) else raw
    return y[0] if isinstance(y, (tuple, list)) else y


class DistillV8Loss:
    """Wraps the stock v8 criterion; adds response distillation."""

    def __init__(self, base, student_model, teacher_model, cfg):
        self.base = base
        self.student = student_model
        self.teacher = teacher_model
        self.cfg = cfg

    def __call__(self, preds, batch):
        loss, loss_items = self.base(preds, batch)
        st = DISTILL
        st["calls"] += 1
        ep = st["epoch"]
        lam = self.cfg["lambda"]
        if self.cfg["ramp"] > 0 and ep >= 0:
            lam *= min(1.0, max(0.0, ep / float(self.cfg["ramp"])))
        if lam <= 0.0:
            return loss, loss_items
        import torch
        import torch.nn.functional as F

        img = batch["img"]
        dev = img.device
        self.student.eval()
        with torch.no_grad():
            t_dec = decoded_output(self.teacher(img)).float()
            s_dec = decoded_output(self.student(img)).float()
        self.student.train()
        if s_dec.shape != t_dec.shape:
            if st["calls"] < 5:
                LOG(f"distill skip: shape mismatch {tuple(s_dec.shape)} vs {tuple(t_dec.shape)}")
            return loss, loss_items
        w = t_dec[:, 4:5].clamp(0.0, 1.0)
        wsum = w.sum().clamp(min=1.0)
        loss_box = F.smooth_l1_loss(
            s_dec[:, :4], t_dec[:, :4], reduction="none"
        ).mean(dim=1, keepdim=True)
        loss_cls = F.mse_loss(s_dec[:, 4:5], w, reduction="none")
        dloss = (loss_box * w + loss_cls * w).sum() / wsum
        if st["calls"] % 200 == 1:
            LOG(f"distill: base={float(loss):.3f} distill={float(dloss):.3f} "
                f"lam={lam:.2f} wsum={float(wsum):.0f}")
        return loss + lam * dloss.to(loss.device), loss_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--name", default="birdnet_d1")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--init", default=None, help="optional .pt warm start (our own weights)")
    ap.add_argument("--from-scratch", action="store_true", help="ignore --init default")
    ap.add_argument("--lambda", dest="lam", type=float, default=2.0)
    ap.add_argument("--ramp", type=int, default=3)
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    cfg = {"lambda": args.lam, "ramp": args.ramp}

    t0 = time.time()
    LOG("=== distill_birdnet.py (response distillation, BirdNet student) ===")
    LOG("args:", vars(args))

    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime", "onnxslim"],
        capture_output=True, text=True,
    )
    LOG("deps install rc:", r.returncode)

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import birdnet

    birdnet.register()
    LOG("modules registered")

    src, from_drive = find_dataset()
    if src is None:
        LOG("FATAL: dataset not found")
        sys.exit(2)
    LOG(f"dataset source: {src} (drive={from_drive})")
    if from_drive and not os.path.exists("/content/data_detection"):
        LOG("copying dataset Drive -> local SSD ...")
        shutil.copytree(src, "/content/data_detection")
        LOG("copy done")
    data_root = "/content/data_detection"
    fix_labels(data_root)
    tr = len(glob.glob(os.path.join(data_root, "train", "images", "*.jpg")))
    va = len(glob.glob(os.path.join(data_root, "val", "images", "*.jpg")))
    LOG(f"train={tr} val={va}")

    import yaml as pyyaml

    with open("/content/bird_data.yaml", "w") as f:
        pyyaml.dump(
            {
                "path": data_root,
                "train": "train/images",
                "val": "val/images",
                "names": {0: "bird"},
            },
            f,
        )

    import torch
    from ultralytics import YOLO

    # teacher ---------------------------------------------------------------
    tpath = find_teacher(args.teacher)
    if tpath is None:
        LOG("FATAL: teacher .pt not found")
        sys.exit(3)
    LOG("teacher:", tpath)
    t_yolo = YOLO(tpath)
    t_model = t_yolo.model.cuda().eval()
    for p in t_model.parameters():
        p.requires_grad_(False)
    try:
        t_model.fuse()
        LOG("teacher fused")
    except Exception as e:
        LOG("teacher fuse skipped:", e)
    with torch.no_grad():
        tprobe = decoded_output(t_model(torch.zeros(1, 3, args.imgsz, args.imgsz).cuda()))
    LOG("teacher probe output:", tuple(tprobe.shape))

    # student ---------------------------------------------------------------
    init = args.init
    if init is None and not args.from_scratch:
        cands = [
            "/content/drive/MyDrive/BIRD/birdnet_v1/best.pt",
            "/content/birdnet_runs/birdnet_v1/weights/best.pt",
        ]
        init = next((c for c in cands if os.path.exists(c)), None)
        if init:
            LOG("warm start from our own v1 artifact:", init)
    if init and os.path.exists(init):
        model = YOLO(init)  # same architecture, our own from-scratch weights
        # reset training state (optimizer/epochs) so this is a fresh run
        model.model.trainer = None
    else:
        model = YOLO(os.path.join(here, "birdnet_v1.yaml"))
        LOG("from-scratch init")
    n_layers, n_params, n_grads, flops = model.info()
    LOG(f"student: {n_layers} layers, {n_params/1e6:.2f}M params")

    # criterion wrap --------------------------------------------------------
    mdl = model.model
    base_init = mdl.init_criterion

    def init_criterion_distill(self):
        return DistillV8Loss(base_init(), self, t_model, cfg)

    mdl.init_criterion = types.MethodType(init_criterion_distill, mdl)
    LOG("criterion wrapped: stock TAL + response distill "
        f"(lambda={cfg['lambda']}, ramp={cfg['ramp']})")

    def on_epoch(trainer):
        DISTILL["epoch"] = trainer.epoch

    model.add_callback("on_train_epoch_start", on_epoch)

    LOG(f"starting training: epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
    model.train(
        data="/content/bird_data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project="/content/birdnet_runs",
        name=args.name,
        exist_ok=True,
        seed=0,
        save_period=50,
        amp=False,  # check_amp() builds a stock yolo26n.yaml; keep as in v1
    )
    LOG("training done")

    best = glob.glob(f"/content/birdnet_runs/{args.name}/weights/best.pt")
    best = best[-1] if best else glob.glob("/content/birdnet_runs/*/weights/best.pt")[-1]
    LOG("validating:", best)
    val_model = YOLO(best)
    metrics = val_model.val(data="/content/bird_data.yaml", split="val", imgsz=args.imgsz)
    result = {
        "weights": best,
        "teacher": tpath,
        "init": init,
        "lambda": args.lam,
        "ramp": args.ramp,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_images": tr,
        "val_images": va,
        "params_m": round(n_params / 1e6, 3),
        "wall_hours": round((time.time() - t0) / 3600, 2),
    }

    if not args.no_export:
        LOG("exporting ONNX @", args.imgsz)
        onnx_path = val_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        LOG("onnx:", onnx_path)
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        ov = sess.run(None, {inp.name: np.zeros((1, 3, args.imgsz, args.imgsz), np.float32)})[0]
        result["onnx"] = onnx_path
        result["onnx_input"] = list(inp.shape)
        result["onnx_output_shape"] = list(ov.shape)
        result["onnx_output_ok"] = bool(len(ov.shape) == 3 and 5 in ov.shape[1:])
        LOG("onnx output shape:", ov.shape, "| layout ok:", result["onnx_output_ok"])

    with open("/content/birdnet_distill_results.json", "w") as f:
        json.dump(result, f, indent=2)
    LOG("RESULT:", json.dumps(result))

    run_dir = os.path.dirname(os.path.dirname(best))
    out = "/content/birdnet_distill_out"
    os.makedirs(out, exist_ok=True)
    for rel in ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"]:
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, os.path.basename(rel)))
    for onx in glob.glob(os.path.join(here, "*.onnx")) + glob.glob("/content/*.onnx"):
        shutil.copy2(onx, out)
    shutil.copy2("/content/birdnet_distill_results.json", out)
    LOG("artifacts in", out)

    if os.path.isdir("/content/drive/MyDrive"):
        dst = f"/content/drive/MyDrive/BIRD/{args.name}"
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(out, "*")):
            shutil.copy2(f, dst)
        LOG("also copied to Drive:", dst)

    LOG(f"=== all done in {result['wall_hours']} h ===")


if __name__ == "__main__":
    main()
