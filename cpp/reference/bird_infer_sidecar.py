#!/usr/bin/env python3
# bird_infer_sidecar.py - persistent YOLOv8 sidecar for birdcpp (C core).
#
# Pipe protocol (binary, little-endian, exact lengths):
#   in:  u32 width, u32 height, width*height*3 raw BGR bytes
#   out: u32 json_len, json bytes, u32 jpg_len, jpg bytes
# The JSON is board_detection.run()'s meta dict (minus annotated_jpg);
# the JPEG is the boxes-drawn frame (or the plain view when nothing was
# detected, like the old python "raw frame is the result" semantics).
#
#  NOTHING else may ever go to stdout - it carries the protocol.
#
# Why a sidecar at all: onnxruntime 1.7.0 on this board ships ONLY as
# the cv2/pybind11 wheel bundle (OrtGetApiBase is not exported, there is
# no plain libonnxruntime.so), so the C core keeps this tiny python
# process as the inference engine. The ONNX forward itself runs in C
# inside cv2.dnn; everything around it (capture, debayer, AWB, turbojpeg,
# HTTP, settings) is C.

import os
import sys
import json
import struct

# die with birdcpp when run outside systemd (manual nohup testing)
try:
    import ctypes
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9, 0, 0, 0)
except Exception:
    pass

BIRD = "/home/root/bird"
if os.path.isdir(BIRD) and BIRD not in sys.path:
    sys.path.insert(0, BIRD)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/root/birdcpp/best.onnx"
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 640
CONF = 0.25
IOU = 0.45

import numpy as np
import cv2
import board_detection as bd

rd = sys.stdin.buffer
wr = sys.stdout.buffer


def emit(meta, jpg):
    body = json.dumps(meta).encode()
    jpg = jpg if jpg else b""
    wr.write(struct.pack("<I", len(body)))
    wr.write(body)
    wr.write(struct.pack("<I", len(jpg)))
    if jpg:
        wr.write(jpg)
    wr.flush()


def main_loop():
    while True:
        hdr = rd.read(8)
        if hdr is None or len(hdr) < 8:
            break
        w, h = struct.unpack("<II", hdr)
        if not (0 < w < 8192 and 0 < h < 8192):
            break
        need = w * h * 3
        buf = rd.read(need)
        while len(buf) < need:
            chunk = rd.read(need - len(buf))
            if not chunk:
                return
            buf += chunk
        img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        try:
            d = bd.run(MODEL, "frame.jpg", IMGSZ, CONF, IOU, verbose=False,
                       bgr_img=img, return_annotated=True)
        except Exception as e:
            d = {"n": None, "infer_ms": None,
                 "err": "detection error: %s" % str(e)[:80]}
            jpg = None
        else:
            jpg = d.pop("annotated_jpg", None)
        if jpg is None:
            # keep /det.jpg updating every cycle: no boxes -> plain view
            ok, enc = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY, 82])
            jpg = enc.tobytes() if ok else None
        emit(d, jpg)


if __name__ == "__main__":
    main_loop()
