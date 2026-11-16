#!/usr/bin/env python
"""train_baseline.py - run the from-scratch baseline on a Colab VM.

Designed to be executed over an SSH tunnel by the PC OpenCode session
(no notebook interaction needed):

    nohup python train_baseline.py > train.log 2>&1 &

Does the full pipeline: locate dataset -> local copy -> label fix -> yaml ->
train yolov8n FROM SCRATCH (blueprint + pretrained=False) -> validate ->
write /content/scratch_results.json -> persist artifacts to /content/scratch_v1
(and to Drive MyDrive/BIRD/scratch_v1 if Drive is mounted).
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
    candidates = [
        "/content/drive/MyDrive/BIRD/data_detection",
        "/content/drive/MyDrive/BIRD/bird_data_detection",
    ]
    for c in candidates:
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
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="yolov8n_scratch_v1")
    ap.add_argument("--drive-save", action="store_true", help="also copy artifacts to Drive")
    args = ap.parse_args()

    t0 = time.time()
    LOG("=== train_baseline.py (FROM SCRATCH) ===")
    LOG("args:", vars(args))

    # 1. dataset ------------------------------------------------------------
    src, from_drive = find_dataset()
    if src is None:
        LOG("FATAL: dataset not found (looked in /content/data_detection and Drive).")
        LOG("Upload data_detection to Drive MyDrive/BIRD/ or scp a zip to /content.")
        sys.exit(2)
    LOG(f"dataset source: {src} (drive={from_drive})")

    if from_drive and not os.path.exists("/content/data_detection"):
        LOG("copying dataset Drive -> local SSD ...")
        shutil.copytree(src, "/content/data_detection")
        LOG("copy done")
    data_root = "/content/data_detection"

    fixed = fix_labels(data_root)
    tr = len(glob.glob(os.path.join(data_root, "train", "images", "*.jpg")))
    va = len(glob.glob(os.path.join(data_root, "val", "images", "*.jpg")))
    LOG(f"train={tr} val={va} labels_fixed={fixed}")

    # strip stray unlabeled artifacts from val (e.g. *_det.jpg renders)
    removed = 0
    for img in glob.glob(os.path.join(data_root, "val", "images", "*")):
        base = os.path.splitext(os.path.basename(img))[0]
        if not os.path.exists(os.path.join(data_root, "val", "labels", base + ".txt")):
            os.remove(img)
            removed += 1
    if removed:
        LOG(f"removed {removed} unlabeled val artifact(s)")

    # 2. yaml ---------------------------------------------------------------
    import yaml

    with open("/content/bird_data.yaml", "w") as f:
        yaml.dump(
            {
                "path": data_root,
                "train": "train/images",
                "val": "val/images",
                "names": {0: "bird"},
            },
            f,
        )

    # 3. deps ---------------------------------------------------------------
    try:
        import ultralytics  # noqa

        LOG("ultralytics already installed:", ultralytics.__version__)
    except ImportError:
        LOG("pip installing ultralytics ...")
        os.system(f"{sys.executable} -m pip install -q ultralytics")

    # 4. train FROM SCRATCH -------------------------------------------------
    from ultralytics import YOLO

    model = YOLO("yolov8n.yaml")  # blueprint only -> random weights
    model.info()
    assert not getattr(model, "ckpt_path", None) or not os.path.exists(
        str(getattr(model, "ckpt_path", ""))
    ), "unexpected pretrained checkpoint attached"

    LOG(f"starting training: epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}")
    model.train(
        data="/content/bird_data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        pretrained=False,  # <- the whole point
        project="/content/scratch_runs",
        name=args.name,
        exist_ok=True,
        seed=0,
        save_period=50,
    )
    LOG("training done")

    # 5. validate -----------------------------------------------------------
    best = glob.glob("/content/scratch_runs/*/weights/best.pt")[-1]
    LOG("validating:", best)
    val_model = YOLO(best)
    metrics = val_model.val(data="/content/bird_data.yaml", split="val")
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
        "wall_hours": round((time.time() - t0) / 3600, 2),
    }
    with open("/content/scratch_results.json", "w") as f:
        json.dump(result, f, indent=2)
    LOG("RESULT:", json.dumps(result))

    # 6. persist ------------------------------------------------------------
    run_dir = os.path.dirname(best)  # .../weights
    run_dir = os.path.dirname(run_dir)
    out = "/content/scratch_v1"
    os.makedirs(out, exist_ok=True)
    for rel in [
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "args.yaml",
        "results.png",
        "confusion_matrix.png",
        "PR_curve.png",
        "F1_curve.png",
    ]:
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out, os.path.basename(rel)))
    for ck in glob.glob(os.path.join(run_dir, "weights", "epoch*.pt")):
        shutil.copy2(ck, out)
    shutil.copy2("/content/scratch_results.json", out)
    LOG("artifacts in", out)

    if args.drive_save and os.path.isdir("/content/drive/MyDrive"):
        dst = "/content/drive/MyDrive/BIRD/scratch_v1"
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(out, "*")):
            shutil.copy2(f, dst)
        LOG("also copied to Drive:", dst)

    LOG(f"=== all done in {result['wall_hours']} h ===")


if __name__ == "__main__":
    main()
