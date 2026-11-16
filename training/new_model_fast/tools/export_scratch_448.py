"""Export the from-scratch baseline (scratch_v1) to ONNX @448 for the board trial.

Mirrors what the production best_448.onnx is: yolov8n exported at imgsz=448.
Writes best_448.onnx next to the .pt in checkpoints/scratch_v1/.

Usage:  venv\Scripts\python.exe training\new_model_fast\tools\export_scratch_448.py
"""
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parents[1]
PT = HERE / "checkpoints" / "scratch_v1" / "best.pt"


def main():
    if not PT.exists():
        raise SystemExit(f"missing {PT}")
    m = YOLO(str(PT))
    out = m.export(format="onnx", imgsz=448, simplify=True)
    print("exported:", out)


if __name__ == "__main__":
    main()
