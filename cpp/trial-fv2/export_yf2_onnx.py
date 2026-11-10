"""Export the COCO-pretrained Yolo-FastestV2 checkpoint to ONNX for the
C trial (cpp/trial-fv2/ -> board /home/root/birdcpp-fv2).

This is a BUILD STEP only: the board never runs python. It produces
one artifact - yf2_coco352.onnx (fixed input 1x3x352x352, two NHWC
outputs [1,22,22,95] stride-16 and [1,11,11,95] stride-32, with
sigmoid(reg)/sigmoid(obj)/softmax(cls) baked into the graph).

Sources live in src-fv2/ (model/, data/, plus both checkpoints) so the
export is reproducible without the original github clone. A stub
torchsummary.py shadows the real package (only summary() is imported
and never used).

Torch 2.x notes handled here:
  * torch.load defaults to weights_only=True -> explicit False.
  * torch.onnx.export may default to the dynamo path -> explicit
    dynamo=False so we get the legacy opset-11 graph the reference
    decode math (utils.handel_preds) was written against.

Usage:  venv python export_yf2_onnx.py    (writes yf2_coco352.onnx here)
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src-fv2"))          # model.*, torchsummary stub

from model.detector import Detector                # noqa: E402

PTH = HERE / "coco2017-0.241078ap-model.pth"
OUT = HERE / "yf2_coco352.onnx"
CLASSES, ANCHOR_NUM, IMGSZ = 80, 3, 352


def main() -> int:
    model = Detector(CLASSES, ANCHOR_NUM, load_param=True, export_onnx=True)
    model.load_state_dict(
        torch.load(PTH, map_location="cpu", weights_only=False))
    model.eval()

    dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)
    try:
        torch.onnx.export(model, dummy, str(OUT),
                          export_params=True, opset_version=11,
                          do_constant_folding=True,
                          input_names=["images"],
                          output_names=["out_grid16", "out_grid32"],
                          dynamic_axes=None, dynamo=False)
    except TypeError:                              # older torch: no dynamo kw
        torch.onnx.export(model, dummy, str(OUT),
                          export_params=True, opset_version=11,
                          do_constant_folding=True,
                          input_names=["images"],
                          output_names=["out_grid16", "out_grid32"])
    mb = OUT.stat().st_size / 1e6
    print(f"exported: {OUT} ({mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    if not PTH.exists():
        sys.exit(f"missing {PTH}")
    sys.exit(main())
