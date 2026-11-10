"""Export the bird YOLOv8-nano model at smaller imgsz for the OPTION-A trial.

The original pipeline runs the model at 640x640; the board trial
(`cpp/trial-a/`, deployed to /home/root/birdcpp-a) tests 448/384
letterboxes which need ~2-3x fewer FLOPs per forward.

Everything happens on the PC (venv has ultralytics, no Colab needed).
Outputs land in cpp/trial-a/best_<sz>.onnx; the production
training/checkpoints/best.onnx (640) is NEVER touched - the export runs
on a temp copy of best.pt.

Board usage afterwards:
  scp cpp/trial-a/* tibird:/home/root/birdcpp-a/
  ssh tibird 'cd /home/root/birdcpp-a && make'
  run with:  /home/root/birdcpp-a/birdcpp-a -p 8890 \
             -m /home/root/birdcpp-a/best_448.onnx --img 448
  (--img must match the size the onnx was exported at.)

Usage:  python export_trial_onnx.py [448 384 ...]   (default 448 384)
"""
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parents[2]      # repo root
PT = HERE / "training" / "checkpoints" / "best.pt"
TRIAL = HERE / "cpp" / "trial-a"


def export(sz: int) -> Path:
    tmp = Path(os.environ["TEMP"]) / f"trialexp{sz}"
    tmp.mkdir(parents=True, exist_ok=True)
    work = tmp / "best.pt"
    shutil.copy(PT, work)
    yolo = YOLO(str(work))
    yolo.export(format="onnx", imgsz=sz)        # writes tmp/best.onnx
    out = TRIAL / f"best_{sz}.onnx"
    shutil.copy(tmp / "best.onnx", out)
    print(f"exported imgsz={sz}: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [448, 384]
    if not PT.exists():
        sys.exit(f"missing {PT}")
    for s in sizes:
        export(s)
