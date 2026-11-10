#!/usr/bin/env python
# fastyolo/infer.py - standalone PC inference checker for the trial
# YOLO-FastestV2 onnx (yf2_coco352.onnx, COCO-pretrained checkpoint,
# same file the board sandbox runs). Decodes the two NHWC heads with
# the reference handel_preds/make_grid math exactly like the C port on
# the board, so what you see here is what the board computes.
#
# IMPORTANT: this export has sigmoid/softmax BAKED into the graph -
# outputs are already activations, do not re-apply them.
#
# Feed matches the reference train/test path AND the board sandbox
# default: BGR /255, PLAIN 352x352 stretch resize (--feed letter for
# letterbox compare).
#
# usage:
#   python infer.py image.jpg [more.jpg ...]     run image(s), draw boxes
#   python infer.py somedir\                     run every image in dir
#   python infer.py --cam 0                      live webcam window
#   flags: --conf 0.25  --iou 0.4  --feed resize|letter
#          --bird-only (class 14 only)  --show-top (draw best cell even
#          below conf)  --out out.png (override output name)
# prints one line per detection and saves <name>_out.jpg next to input.
import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

IMG = 352
STRIDES = (16, 32)
# anchors from the reference coco.data: trio 0 -> stride-16 grid,
# trio 1 -> stride-32 grid (same numbers as FV2_ANCH in birdcpp.c)
ANCHORS = (
    (12.64, 19.39, 37.88, 51.48, 55.71, 138.31),
    (126.91, 78.23, 131.57, 214.55, 279.92, 258.87),
)
CH = 95          # per-cell channels: reg 3*4 | obj 3 | cls 80
OBJ_OFF = 12
CLS_OFF = 15
BIRD_CLS = 14


def load_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
        if len(names) == 80:
            return names
    except OSError:
        pass
    print("coco.names not found/incomplete - using numeric cls labels")
    return ["cls%d" % i for i in range(80)]


def letterbox_or_resize(img, feed):
    # both return BGR/255 float CHW 1x3x352x352; resize = plain stretch
    h, w = img.shape[:2]
    if feed == "resize":
        r = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        x = r.astype(np.float32) / 255.0
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), (0, 0, 1.0, 1.0)
    # letterbox: scale to fit, pad bottom/right with 114 (yolov5 style)
    scale = min(IMG / h, IMG / w)
    nh, nw = round(h * scale), round(w * scale)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMG, IMG, 3), 114, np.uint8)
    canvas[:nh, :nw] = r
    x = canvas.astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), (0, 0, scale, scale)


def decode_heads(out16, out32, conf, iou, bird_only, show_top):
    # outputs are NHWC [1,g,g,95]; undo the reference anchor decode:
    # bx=(tx*2-0.5+gx)*stride  bw=(tx*2)^2*anchor_w  (t already activated)
    cands = []  # (x1,y1,x2,y2,score,cls)
    for stride, grid, anchors in ((STRIDES[0], out16, ANCHORS[0]),
                                  (STRIDES[1], out32, ANCHORS[1])):
        g = grid.shape[1]
        flat = grid.reshape(-1, CH)
        obj = flat[:, OBJ_OFF:OBJ_OFF + 3]          # (N,3) sigmoided
        cls = flat[:, CLS_OFF:CLS_OFF + 80]          # (N,80) softmaxed
        if bird_only:
            best_cls = cls[:, BIRD_CLS:BIRD_CLS + 1]
            best_idx = np.full((len(flat), 1), BIRD_CLS, np.int64)
        else:
            best_idx = cls.argmax(axis=1, keepdims=True)
            best_cls = np.take_along_axis(cls, best_idx, axis=1)
        score = obj * best_cls                       # (N,3)
        gy, gx = np.divmod(np.arange(len(flat)), g)
        for a in range(3):
            m = score[:, a] > conf
            if not m.any():
                continue
            idx = np.where(m)[0]
            reg = flat[idx, a * 4:a * 4 + 4]
            aw, ah = anchors[a * 2], anchors[a * 2 + 1]
            bx = (reg[:, 0] * 2.0 - 0.5 + gx[idx]) * stride
            by = (reg[:, 1] * 2.0 - 0.5 + gy[idx]) * stride
            bw = (reg[:, 2] * 2.0) ** 2 * aw
            bh = (reg[:, 3] * 2.0) ** 2 * ah
            for j in range(len(idx)):
                cands.append((bx[j] - bw[j] / 2, by[j] - bh[j] / 2,
                              bx[j] + bw[j] / 2, by[j] + bh[j] / 2,
                              float(score[idx[j], a]), int(best_idx[idx[j], 0])))
    # greedy class-agnostic NMS (matches the C sandbox behavior)
    cands.sort(key=lambda c: -c[4])
    keep = []
    for c in cands:
        ok = True
        for k in keep:
            xx1, yy1 = max(c[0], k[0]), max(c[1], k[1])
            xx2, yy2 = min(c[2], k[2]), min(c[3], k[3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            a1 = (c[2] - c[0]) * (c[3] - c[1])
            a2 = (k[2] - k[0]) * (k[3] - k[1])
            union = a1 + a2 - inter
            if union > 0 and inter / union > iou:
                ok = False
                break
        if ok:
            keep.append(c)
        if len(keep) >= 64:
            break
    if not keep and show_top:
        # best cell of the whole head regardless of conf, like --showtop
        best = None
        for stride, grid, anchors in ((STRIDES[0], out16, ANCHORS[0]),
                                      (STRIDES[1], out32, ANCHORS[1])):
            g = grid.shape[1]
            flat = grid.reshape(-1, CH)
            s = flat[:, OBJ_OFF:OBJ_OFF + 3] * flat[:, CLS_OFF:CLS_OFF + 80].max(axis=1, keepdims=True)
            i = int(s.argmax())
            gy, gx = divmod(i, g)
            a = int(s[i].argmax())
            reg = flat[i, a * 4:a * 4 + 4]
            aw, ah = anchors[a * 2], anchors[a * 2 + 1]
            bx = (reg[0] * 2.0 - 0.5 + gx) * stride
            by = (reg[1] * 2.0 - 0.5 + gy) * stride
            bw = (reg[2] * 2.0) ** 2 * aw
            bh = (reg[3] * 2.0) ** 2 * ah
            best = (bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2,
                    float(s[i, a]), BIRD_CLS if bird_only else
                    int(flat[i, CLS_OFF:CLS_OFF + 80].argmax()))
        if best:
            keep.append(best)
    return keep


def draw(img, dets, names, scale, ox, oy):
    for x1, y1, x2, y2, sc, c in dets:
        # map model-space back to source image space
        px1 = int((x1 - ox) / scale); py1 = int((y1 - oy) / scale)
        px2 = int((x2 - ox) / scale); py2 = int((y2 - oy) / scale)
        h, w = img.shape[:2]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w - 1, px2), min(h - 1, py2)
        label = "%s %.3f" % (names[c] if 0 <= c < len(names) else "cls%d" % c, sc)
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        ly = py1 - th - 6 if py1 - th - 6 > 0 else py1 + th + 6
        cv2.rectangle(img, (px1, ly - th - 4), (px1 + tw + 4, ly + 4), (0, 0, 0), -1)
        cv2.putText(img, label, (px1 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def run_image(sess, names, path, args):
    img = cv2.imread(path)
    if img is None:
        print("skip (unreadable):", path)
        return
    x, (ox, oy, sx, sy) = letterbox_or_resize(img, args.feed)
    out16, out32 = sess.run(None, {sess.get_inputs()[0].name: x})
    t0 = cv2.getTickCount()
    out16, out32 = sess.run(None, {sess.get_inputs()[0].name: x})
    ms = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000.0
    dets = decode_heads(out16, out32, args.conf, args.iou, args.bird_only,
                        args.show_top)
    # letterbox pads are bottom/right only -> same offsets for map-back
    print("%s  [fwd %.0f ms]" % (os.path.basename(path), ms))
    if not dets:
        print("  no detections > conf %g" % args.conf)
    for x1, y1, x2, y2, sc, c in dets:
        print("  %-14s %.3f  (%d,%d)-(%d,%d)" %
              (names[c] if 0 <= c < len(names) else "cls%d" % c, sc,
               (x1 - ox) / sx, (y1 - oy) / sy, (x2 - ox) / sx, (y2 - oy) / sy))
    vis = draw(img.copy(), dets, names, sx, ox, oy)
    root, ext = os.path.splitext(path)
    outp = args.out or (root + "_out.jpg")
    cv2.imwrite(outp, vis)
    print("  saved:", outp)


def run_cam(sess, names, args):
    cam = cv2.VideoCapture(args.cam)
    if not cam.isOpened():
        sys.exit("cannot open camera %d" % args.cam)
    print("webcam live - q quits")
    while True:
        ok, img = cam.read()
        if not ok:
            break
        x, (ox, oy, sx, sy) = letterbox_or_resize(img, args.feed)
        out16, out32 = sess.run(None, {sess.get_inputs()[0].name: x})
        dets = decode_heads(out16, out32, args.conf, args.iou,
                            args.bird_only, args.show_top)
        vis = draw(img, dets, names, sx, ox, oy)
        cv2.putText(vis, "conf %g iou %g feed %s" %
                    (args.conf, args.iou, args.feed), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("fastyolo", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="fastyolo fv2 inference checker")
    ap.add_argument("inputs", nargs="*", help="image files or a directory")
    ap.add_argument("--cam", type=int, default=-1, help="webcam index (live)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.40)
    ap.add_argument("--feed", choices=("resize", "letter"), default="resize")
    ap.add_argument("--bird-only", action="store_true",
                    help="decode class 14 (bird) only, like production")
    ap.add_argument("--show-top", action="store_true",
                    help="if nothing passes conf, draw the best cell anyway")
    ap.add_argument("--out", default=None, help="output path (single image)")
    args = ap.parse_args()

    model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "yf2_coco352.onnx")
    names = load_names(os.path.join(os.path.dirname(model), "coco.names"))
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("model:", model, "| input:", inp.name, inp.shape)

    if args.cam >= 0:
        run_cam(sess, names, args)
        return
    paths = []
    for p in args.inputs:
        if os.path.isdir(p):
            paths += [os.path.join(p, f) for f in sorted(os.listdir(p))
                      if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        else:
            paths.append(p)
    if not paths:
        sys.exit("nothing to run - give an image/dir or --cam N")
    for p in paths:
        run_image(sess, names, p, args)


if __name__ == "__main__":
    main()
