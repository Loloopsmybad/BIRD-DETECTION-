FASTYOLO - PC-side inference checker for the trial YOLO-FastestV2
=================================================================
Standalone folder: model + labels + one python script. Runs the SAME
onnx the board sandbox ran (yf2_coco352.onnx, COCO-pretrained) with
the SAME decode math as the board C (handel_preds/make_grid, anchors
from coco.data, class-agnostic greedy NMS), so what you see here is
what the board computed.

Files
-----
  yf2_coco352.onnx   the exported model (1 MB, fixed 352x352 input)
  coco.names         the 80 COCO class labels (from the model repo)
  infer.py           the checker (needs: onnxruntime, opencv, numpy)
  24_647.jpg         val-split sample image for a quick smoke test

Usage (from the BIRD-DETECTION venv)
------------------------------------
  venv\Scripts\python.exe fastyolo\infer.py fastyolo\24_647.jpg
  venv\Scripts\python.exe fastyolo\infer.py somepic.jpg --conf 0.25
  venv\Scripts\python.exe fastyolo\infer.py somedir\          (batch)
  venv\Scripts\python.exe fastyolo\infer.py --cam 0           (webcam,
                                     live window, q quits)

Flags: --conf 0.25 (min score)  --iou 0.40 (NMS)  --feed resize|letter
       --bird-only (class 14 only)  --show-top (best cell even below
       conf)  --out path.png
Output: detections printed + <name>_out.jpg with boxes drawn.

Verified parity (identical pixels)
----------------------------------
val image 24_647.jpg, resize feed: PC script top bird 0.762 vs native
PyTorch 0.768 / board C 0.768 on the same file - agreement within the
known 3rd-decimal scaler tolerance (each implementation's resize
differs microscopically; the chain native<->onnx<->board-C<->PC is
consistent).

Notes
-----
* The export has sigmoid/softmax baked in - the script uses raw head
  outputs directly (no re-activation), same as the board C.
* Feed default is the reference plain stretch resize (measured better
  than letterbox on this checkpoint); --feed letter to compare.
* Expect the COCO-pretrained accuracy limits: loose/offset boxes and
  misses on small objects (AP~0 on our feeder-bird val split vs 0.806
  for production yolov8n@448). This tool is for inspecting inference,
  not a production detector.
* fwd on a modern PC CPU: ~2-3 ms (vs ~120 ms on the board).
