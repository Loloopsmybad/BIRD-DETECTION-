#!/usr/bin/env python3
# board_detection.py — YOLOv8 ONNX detection runner for the SK-AM62 board.
# Runs with only numpy + opencv (no torch/ultralytics needed on the board).
#
# Usage:
#   python3 board_detection.py <image> [image2 ...] [--save] [--conf 0.25] [--iou 0.45] [--imgsz 640]
#
# Prints detections as: <file> ['] bird <conf> (x1,y1,x2,y2) and saves
# annotated copies to *_det.jpg when --save is given.

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

CLASS_NAME = "bird"

# Loaded ONNX models are cached by path: the first call pays the load cost,
# every later call in the same process skips re-reading/re-parsing the model.
_NET = {}


def _nms(b_xywh, scs, conf_thr, iou_thr):
    """Greedy NMS in pure numpy. b_xywh: Nx4 float (x, y, w, h)."""
    if len(b_xywh) == 0:
        return []
    x1 = b_xywh[:, 0]
    y1 = b_xywh[:, 1]
    x2 = x1 + b_xywh[:, 2]
    y2 = y1 + b_xywh[:, 3]
    order = scs.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        area_i = max(1e-9, b_xywh[i, 2] * b_xywh[i, 3])
        area_o = np.maximum(1e-9, b_xywh[order[1:], 2] * b_xywh[order[1:], 3])
        iou = inter / (area_i + area_o - inter)
        order = order[1:][iou <= iou_thr]
    return keep


def letterbox(img, new=640):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), 114, dtype=np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, top, left


def run(net_path, img_path, imgsz=640, conf_thr=0.25, iou_thr=0.45, save=False,
        verbose=True, jpeg_bytes=None):
    """Run detection. Returns a dict (or dict with "err" on failure).

    jpeg_bytes: pass raw JPEG bytes to decode directly from memory instead
    of reading a file (saves a disk write+read per frame in server mode).
    verbose=False keeps stdout clean when used as an imported module.
    """
    if not os.path.exists(net_path):
        if verbose:
            print("MODEL_NOT_FOUND:", net_path)
        return {"err": "MODEL_NOT_FOUND: %s" % net_path}
    if jpeg_bytes is None and not os.path.exists(img_path):
        if verbose:
            print("IMAGE_NOT_FOUND:", img_path)
        return {"err": "IMAGE_NOT_FOUND: %s" % img_path}

    net = _NET.get(net_path)
    if net is None:
        net = cv2.dnn.readNetFromONNX(net_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _NET[net_path] = net
    if jpeg_bytes is not None:
        img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_path)
    if img is None:
        if verbose:
            print("IMAGE_DECODE_FAIL:", img_path)
        return {"err": "IMAGE_DECODE_FAIL: %s" % img_path}
    h, w = img.shape[:2]

    blob_in, r, top, left = letterbox(img, imgsz)
    blob = cv2.dnn.blobFromImage(blob_in, 1.0 / 255.0, (imgsz, imgsz),
                                 swapRB=True, crop=False)

    t0 = time.time()
    net.setInput(blob)
    out = net.forward()
    infer_ms = (time.time() - t0) * 1000.0

    # YOLOv8 export shape: (1, 4+nc, N). Normalize to (4+nc, N) rows=channels.
    out = np.squeeze(out)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.shape[0] > out.shape[1]:
        out = out.T                      # (N, 4+nc) -> (4+nc, N)
    nc = out.shape[0] - 4
    bbox = out[0:4, :]                # cx, cy, w, h
    cls = out[4:4 + nc, :]            # class scores (nc rows)
    scores = cls.max(axis=0) if nc > 1 else cls[0]

    keep = scores > conf_thr
    boxes_raw, scs = bbox[:, keep].T, scores[keep]

    results = []
    if len(boxes_raw):
        # xywh center -> xyxy, then undo letterbox
        b = boxes_raw.copy()
        b[:, 0] -= b[:, 2] / 2
        b[:, 1] -= b[:, 3] / 2
        for i in range(b.shape[0]):
            cx, cy = b[i, 0:2] - [left, top]
            x1, y1 = cx / r, cy / r
            x2, y2 = (cx + b[i, 2]) / r, (cy + b[i, 3]) / r
            results.append([x1, y1, x2, y2, float(scs[i])])

        keep_idx = _nms(b, scs, conf_thr, iou_thr)
        if keep_idx:
            results = [results[i] for i in keep_idx]

    h_img = img.shape[0]
    th = max(3, h_img // 240)
    fs = max(0.55, h_img / 1200.0)
    tf = max(1, int(fs * 2))
    for x1, y1, x2, y2, s in results:
        xi, yi = max(0, int(x1)), max(0, int(y1))
        xa, ya = max(0, int(x2)), max(0, int(y2))
        label = "%s %.2f" % (CLASS_NAME, s)
        tw, tth = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, tf)[0]
        ly = yi - tth - 8
        if ly < 0:
            ly = yi + tth + 10
        cv2.rectangle(img, (xi, ly - tth - 6), (xi + tw + 8, ly + 6), (0, 0, 0), -1)
        cv2.rectangle(img, (xi, yi), (xa, ya), (0, 255, 0), th)
        cv2.putText(img, label, (xi + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 255, 0), tf, cv2.LINE_AA)

    det_lines = ["bird %.2f (%d,%d,%d,%d)" % (s, int(x1), int(y1), int(x2), int(y2))
                 for x1, y1, x2, y2, s in results]
    meta = {"file": img_path, "n": len(results),
            "infer_ms": round(infer_ms, 1), "size": [w, h],
            "dets": det_lines}
    if save:
        ext = os.path.splitext(img_path)[1]
        out_path = img_path.replace(ext, "_det" + ext)
        if results or jpeg_bytes is None:
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        else:
            # nothing to draw: keep the original JPEG bytes untouched
            # (skips re-encoding a frame with no detections)
            with open(out_path, "wb") as f:
                f.write(jpeg_bytes)
    if verbose:
        print(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", type=str, default="best.onnx")
    args = ap.parse_args()

    for p in args.images:
        run(args.model, p, args.imgsz, args.conf, args.iou, args.save)


if __name__ == "__main__":
    main()
