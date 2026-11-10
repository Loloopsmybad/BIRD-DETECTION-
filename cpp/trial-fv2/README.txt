YOLO-FASTESTV2 TRIAL - ultra-light head vs the production YOLOv8-nano
=====================================================================
Folder cpp\trial-fv2\ (this folder) <-> board /home/root/birdcpp-fv2/.
SANDBOX next to production (/home/root/birdcpp, port 8888): own binary
birdcpp-fv2, own port 8890, started by hand, no systemd unit. When the
trial is NOT running, production is untouched and owns the camera.

WHAT THIS TRIAL IS
------------------
The user asked to try Yolo-FastestV2 (dog-qiuqiu, 2022) instead of the
current model. Sizes: 0.25 M params / 0.212 GFLOPs / 1.0 MB onnx vs the
production YOLOv8-nano (~3.2 M params / ~8.7 GFLOPs @448 / 12 MB onnx).
No retraining: the trial uses the COCO-pretrained checkpoint
(coco2017-0.241078ap-model.pth, mAP50 0.241 COCO@352 overall) which has
a native "bird" class (COCO index 14).

EVERYTHING RUNTIME IS C (user requirement: extract every last drop):
* birdcpp.c   - production birdcpp + fv2 head: two-output ORT run,
  preallocated persistent outputs (zero per-cycle allocation), the
  handel_preds/make_grid decode in C, greedy NMS reuse, --fv2 / --conf
  / --iou / --feed flags, birdcpp-fv2 journal identity, det-stage
  timers inherited.
* eval.c      - the SAME decode as a file-driven harness (no camera):
  scans DATADIR/images + labels, computes AP / P / R / F1 in C, runs
  on the board (production ORT + turbojpeg).
* export_yf2_onnx.py - the ONLY python, a one-time PC build step that
  converts the .pth into yf2_coco352.onnx (opset 11, sigmoid/softmax
  baked into the graph, fixed 1x3x352x352, two NHWC outputs
  out_grid16 [1,22,22,95] + out_grid32 [1,11,11,95]).

HEAD LAYOUT (decoded from the reference sources, verified bit-level)
--------------------------------------------------------------------
channels [reg 3*4 | obj 3 | cls 80] per cell; reg/obj sigmoided and
cls softmaxed INSIDE the exported graph:
    ch[a*4+0..3] = anchor-a tx,ty,tw,th
    ch[12+a]     = objectness-a
    ch[15+c]     = class-c probability (bird = 14)
decode (reference utils.handel_preds):
    bx=(sig(tx)*2-0.5+gx)*stride   by=(sig(ty)*2-0.5+gy)*stride
    bw=(sig(tw)*2)^2*anchor_w      bh=(sig(th)*2)^2*anchor_h
anchors (coco.data): trio 12.64,19.39 37.88,51.48 55.71,138.31 for the
stride-16 grid; 126.91,78.23 131.57,214.55 279.92,258.87 for stride-32.
FEED: the reference train/test path keeps cv2 BGR and /255, and feeds a
PLAIN 352x352 STRETCH RESIZE (no letterbox). Measured impact below.

VERIFICATION TRAIL (why we trust the C port)
--------------------------------------------
* PC diag (diag_native_vs_onnx.py): native torch forward vs ONNX
  runtime output on the same tensor: max diff 1.4e-6 (obj), 2.8e-6
  (cls14) -> export is exact.
* Board eval probe (--probe-name, --feed resize): top cells match the
  PC native numbers to 3 decimals (obj 0.768/0.775/0.781 vs
  0.768/0.772/0.783 on val image 24_647.jpg) -> C letterbox/resize +
  decode + ORT on the board are correct.
* The earlier low-score mystery was the FEED: letterbox costs ~2x
  objectness vs the reference plain-resize (trained distribution).

SPEED (live camera, TI SK-AM62, stage timers, production STOPPED)
------------------------------------------------------------------
| config                          | fwd avg | cycle  | det rate |
|---------------------------------|---------|--------|----------|
| fv2@352 resize-feed, 4 threads  | ~120 ms | ~142 ms | ~7 Hz   |
| (1-thread run: measured, see note)                           |
| production yolov8n@448, 4 thr   | ~875 ms | ~915 ms | ~1.1 Hz |

(1-thread note: the --ort-threads 1 measurement window was cut short by
a board Wi-Fi flake; rerun: birdcpp-fv2 -p 8890 -m yf2_coco352.onnx
--fv2 --feed resize --ort-threads 1, then poll /phone stage.)

ACCURACY (board, eval.c on the 191-image val split, 413 birds; same
tight GT labels that give yolov8n its numbers; fwd numbers here are
contention-inflated because production birdcpp kept running)
------------------------------------------------------------------
| feed      | match IoU | AP    | P@0.05 | R@0.05 |
|-----------|-----------|-------|--------|--------|
| resize    | 0.50      | 0.000 | 0.011  | 0.019  |
| resize    | 0.30      | 0.008 | 0.061  | 0.109  |
| letterbox | 0.50      | 0.001 | 0.016  | 0.022  |
For comparison (ultralytics val, IoU 0.5, same split):
yolov8n custom-trained = 0.832 @640 / 0.806 @448.

VERDICT
-------
The COCO-pretrained fv2 detects birds (obj*cls14 up to ~0.76 on val
bird scenes; probe shows boxes ON birds) but its boxes are LOOSE and
offset vs our tight labels - it was never trained on our feeder birds
- so AP@0.5 collapses to ~0. The production yolov8n keeps its ~0.8.
fv2 is only viable after RETRAINING on our dataset (then re-export via
export_yf2_onnx.py with the same bake-in; the C decode stays identical,
only the anchor numbers would come from the new training cfg).
Speed though is spectacular: ~120 ms fwd = ~7 detections/s and tiny
CPU load - if a retrained fv2 ever reaches acceptable mAP, the same
sandbox binary is the drop-in runtime.

LIVE SANDBOX RUN (production stopped for the window)
----------------------------------------------------
  ssh tibird 'systemctl stop birdcpp; killall -9 pumper
              cd /home/root/birdcpp-fv2
              nohup ./birdcpp-fv2 -p 8890 \
                -m /home/root/birdcpp-fv2/yf2_coco352.onnx \
                --fv2 --feed resize --conf 0.3 >/tmp/trialFV2.log 2>&1 &'
  dashboard: http://192.168.1.22:8890/   (stage block in /phone)
  restore:   ssh tibird 'killall birdcpp-fv2; killall -9 frame_pumper
                         systemctl start birdcpp'
FLAGS: --fv2 (two-output head; pins --img to the fixed 352 export),
--conf (default 0.30 reference), --iou (default 0.40 reference),
--feed resize|letter (fv2 only; default letterbox - resize measured
better), --ort-threads N (default 4), --showtop (when nothing clears
conf, draw the head's best cell with its true score - always shows
what the head is "looking at"), --allcls (decode ALL 80 COCO classes,
not just bird=14: per-anchor obj prune, then argmax over the 80
softmaxed probs; boxes labeled with the real class name).

ALL-80-CLASS MODE (--allcls) - verified live
--------------------------------------------
The COCO checkpoint natively detects all 80 classes; labels ship with
the repo (src-fv2\data\coco.names, scp'd to the board next to the
binary; a built-in copy in birdcpp.c is the fallback). Live proof:
person 0.44-0.71 boxed + labeled correctly on the dashboard stream;
detbox_t carries .cls, meta.json dets text uses the class name, and
the 5x7 label font grew the full lowercase alphabet (it previously
had exactly b,d,i,r - "person" rendered as a lone 'r' glyph!). NMS is
class-agnostic here (fine for visualization). Bird-only behavior is
unchanged without the flag; production mode always decodes nc=1 bird.

OPS NOTES (camera-exclusivity fights, learned the hard way)
-----------------------------------------------------------
* The orphan copier cure is `killall -9 frame_pumper` (NOT `pumper` -
  the binary is /home/root/bird*/frame_pumper; killall by the old name
  silently finds nothing). Symptom: "VIDIOC_S_FMT: Device or resource
  busy" / "sensor apply rejected: horizontal_flip: busy" retry loops.
* The legacy python birdboard (board_camera_server.py, birdboard
  service) auto-started at boot and stole the camera behind
  production's back - a res-switch in the UI exposed it as a freeze
  (the trial lost the tug-of-war on every camera re-open). Fixed by
  `systemctl disable birdboard` (re-enable with `systemctl enable
  birdboard` if the old board is ever wanted). After eviction, mode
  switches 1640x1232 <-> 1920x1080 were reproduced clean via POST
  /camera-set: buffers rebuild, detection keeps flowing, no EBUSY.

OFFLINE EVAL (no camera needed, runs while production is up)
------------------------------------------------------------
  scp -r training\data_detection\val tibird:/home/root/fv2val
  ./yf2eval -m yf2_coco352.onnx -d /home/root/fv2val \
            [--feed resize] [--ap-iou 0.3] [--probe-name X.jpg]

FILES: birdcpp.c, eval.c, Makefile, yf2_coco352.onnx,
coco2017-0.241078ap-model.pth, src-fv2\ (upstream model/, data/ with
coco.names - the 80-class label source), export_yf2_onnx.py,
diag_native_vs_onnx.py, fv2*.jpg (sample sandbox stream frames).

NOTES
-----
* /det.jpg is 410 by design (live-media-only); annotated frames are in
  /stream. Sample frames above were extracted from a stream capture.
* The upstream clone this was copied from: github.com/dog-qiuqiu/
  Yolo-FastestV2 (modelzoo checkpoint included here for reproducibility).
* Keep eval.c decode + birdcpp.c fv2_decode in sync if one changes.
