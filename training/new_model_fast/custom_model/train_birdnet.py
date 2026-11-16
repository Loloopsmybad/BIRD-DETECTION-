#!/usr/bin/env python
"""train_birdnet.py - train BirdNet v1 (custom ReLU-only architecture) from scratch.

Runs on the Colab VM (driven over colab-mcp):

    python train_birdnet.py > birdnet.log 2>&1 &

Pipeline: register modules -> dataset find/copy/fix -> yaml -> train from
scratch (imgsz 448 = deployment size) -> validate -> export ONNX (1,5,N) ->
verify ONNX shape with onnxruntime -> write /content/birdnet_results.json ->
persist artifacts to /content/birdnet_v1_out and Drive MyDrive/BIRD/birdnet_v1.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

LOG = lambda *a: print(*a, flush=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--name", default="birdnet_v1")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    LOG("=== train_birdnet.py (custom ReLU-only BirdNet) ===")
    LOG("args:", vars(args))

    # 0. deps ---------------------------------------------------------------
    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime", "onnxslim"],
        capture_output=True, text=True,
    )
    LOG("deps install rc:", r.returncode)

    # 1. register custom modules BEFORE building ----------------------------
    # birdnet.py + the yaml are uploaded next to this script on the VM.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import birdnet

    birdnet.register()
    LOG("modules registered (BConv/BC2f/BirdSPPF/BirdDetect)")

    # 2. dataset ------------------------------------------------------------
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

    # strip stray unlabeled artifacts from val
    removed = 0
    for img in glob.glob(os.path.join(data_root, "val", "images", "*")):
        base = os.path.splitext(os.path.basename(img))[0]
        if not os.path.exists(os.path.join(data_root, "val", "labels", base + ".txt")):
            os.remove(img)
            removed += 1
    if removed:
        LOG(f"removed {removed} unlabeled val artifact(s)")

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

    # 3. build custom model -------------------------------------------------
    from ultralytics import YOLO

    yaml_path = os.path.join(here, "birdnet_v1.yaml")
    model = YOLO(yaml_path)  # blueprint only -> random weights
    info = model.info()      # (layers, params, gradients, flops)
    n_layers, n_params, n_grads, flops = info
    LOG(f"model: {n_layers} layers, {n_params/1e6:.2f}M params, {flops:.2f} GFLOPs @640")

    # 4. train FROM SCRATCH -------------------------------------------------
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
    )
    LOG("training done")

    # 5. validate (deployment size) -----------------------------------------
    best = glob.glob("/content/birdnet_runs/*/weights/best.pt")[-1]
    LOG("validating:", best)
    val_model = YOLO(best)
    metrics = val_model.val(data="/content/bird_data.yaml", split="val", imgsz=args.imgsz)
    result = {
        "weights": best,
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
        "gflops_at_640": round(float(flops), 3) if flops else None,
        "wall_hours": round((time.time() - t0) / 3600, 2),
    }

    # 6. export ONNX + verify shape -----------------------------------------
    if not args.no_export:
        LOG("exporting ONNX @", args.imgsz)
        onnx_path = val_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        LOG("onnx:", onnx_path)
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        out = sess.get_outputs()[0]
        h = w = args.imgsz
        dummy = torch_dummy(w, h)
        ov = sess.run(None, {inp.name: dummy})[0]
        result["onnx"] = onnx_path
        result["onnx_input"] = list(inp.shape)
        result["onnx_output_shape"] = list(ov.shape)
        result["onnx_output_ok"] = bool(
            len(ov.shape) == 3 and (ov.shape[1] == 5 or ov.shape[2] == 5)
        )
        LOG("onnx output shape:", ov.shape, "| layout ok:", result["onnx_output_ok"])

    with open("/content/birdnet_results.json", "w") as f:
        json.dump(result, f, indent=2)
    LOG("RESULT:", json.dumps(result))

    # 7. persist ------------------------------------------------------------
    run_dir = os.path.dirname(os.path.dirname(best))
    out = "/content/birdnet_v1_out"
    os.makedirs(out, exist_ok=True)
    for rel in ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"]:
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, os.path.basename(rel)))
    for onx in glob.glob(os.path.join(here, "*.onnx")) + glob.glob("/content/*.onnx"):
        shutil.copy2(onx, out)
    shutil.copy2("/content/birdnet_results.json", out)
    LOG("artifacts in", out)

    if os.path.isdir("/content/drive/MyDrive"):
        dst = "/content/drive/MyDrive/BIRD/birdnet_v1"
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(out, "*")):
            shutil.copy2(f, dst)
        LOG("also copied to Drive:", dst)

    LOG(f"=== all done in {result['wall_hours']} h ===")


def torch_dummy(w, h):
    import numpy as np

    return np.zeros((1, 3, h, w), dtype=np.float32)


if __name__ == "__main__":
    main()
