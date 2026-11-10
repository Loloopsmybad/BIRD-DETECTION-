OPTION-A TRIAL - smaller model letterbox on the board (birdcpp-a)
=================================================================
Folder cpp\trial-a\ (this folder) <-> board /home/root/birdcpp-a/.
It is a SANDBOX next to production (/home/root/birdcpp, port 8888):
* birdcpp.c here = production cpp\birdcpp.c with ONE change: the model
  letterbox size became a RUNTIME flag (--img N, multiple of 32,
  320..640; default 640) instead of a #define, plus the journal prefix
  printed as "birdcpp-a:".
* Models: best_448.onnx / best_384.onnx exported LOCALLY on the PC by
  training\tools\export_trial_onnx.py (from training\checkpoints\best.pt,
  temp copy - the production best.onnx is never touched).
  _448 output = 1x5x4116 candidates   (784... 56^2+28^2+14^2 = 4116)
  _384 output = 1x5x3024 candidates
* Binary name is birdcpp-a (own Makefile) so `killall` can't hurt
  production; the trial is started by hand, no systemd unit.

RUN (takes the camera exclusively - production birdcpp must be STOPPED
while the trial runs, restarted afterwards):
  ssh tibird 'systemctl stop birdcpp; killall -9 pumper 2>/dev/null
              cd /home/root/birdcpp-a
              nohup ./birdcpp-a -p 8890 \
                 -m /home/root/birdcpp-a/best_448.onnx --img 448 \
                 >/tmp/trialA.log 2>&1 &'
  trial dashboard: http://192.168.1.22:8890/   (stage timings in /phone)
  swap size: killall birdcpp-a first, then relaunch with the other
  --img/-m pair (they must MATCH: --img is the exported imgsz).
  restore:    ssh tibird 'killall birdcpp-a; killall -9 pumper
                          systemctl start birdcpp'

MEASURED on the live camera (2026-11-10, 8 /phone samples each,
stage timers from the 2026-11-10 instrumentation; all runs on the
OPTION-B core partition - copier parked on core 3, 1 thread, nice 10):

| model (imgsz, letterbox) | fwd avg | cycle avg | notes |
|---|---|---|---|
| 640 + opt-B (current PRODUCTION) | ~1735 ms | ~1810 ms | ground truth accuracy |
| 448 + opt-B (best_448.onnx)      | ~1053 ms | ~1094 ms | ~40% faster cycles |
| 384 + opt-B (best_384.onnx)      | ~717 ms  | ~747 ms  | ~2.4x the prod forward; millisecond-scale letterbox (~25 ms) |

Context: the ORIGINAL live wall (before opt-B) was fwd ~2740 / cycle
~2830 ms. Quiesced 640 forward remains the documented ~1150 ms.

ACCURACY VALIDATED (2026-11-10, yolo val on the 191-image val split,
413 bird instances, run on the PC from THESE exact onnx artifacts):

| imgsz        |  P     |  R     | mAP50  | mAP50-95 | vs 640        |
|--------------|--------|--------|--------|----------|---------------|
| 640 baseline | 0.797  | 0.753  | 0.832  | 0.662    | -             |
| 448 trial    | 0.774  | 0.741  | 0.806  | 0.587    | -3.1% mAP50   |
| 384 trial    | 0.768  | 0.705  | 0.767  | 0.557    | -7.8% mAP50   |

Recall (R) = share of actual birds found; 384 misses ~6% more birds.
Read: 448 is the balanced promotion candidate; 384 is the speed-first
option. Dataset yaml: training\data_detection\bird_val.yaml.
