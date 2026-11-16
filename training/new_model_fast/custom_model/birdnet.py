"""BirdNet v1 - lightweight single-class bird detector, trained from scratch.

Why custom modules
------------------
Board reality: TI SK-AM62, 4x Cortex-A53, ONNX Runtime CPU (NEON).
Ultralytics' default act is SiLU = x*sigmoid(x) -> mul+sigmoid+mul elementwise;
ReLU is a single max(). On a ~90 GFLOPS NEON ceiling that difference is real
money at the elementwise layers. So every conv here is conv-bn-ReLU.

How it plugs into ultralytics
-----------------------------
The training stack (TAL assigner, DFL loss, mosaic, EMA, validator) is
excellent and we want it for free. The yaml therefore uses the STANDARD
module names (Conv, C2f, SPPF, Detect), and register() monkey-patches them in
ultralytics.nn.tasks so parse_model resolves those names to the classes below.
This keeps parse_model's channel plumbing and stride computation untouched,
and BirdDetect subclasses the real Detect so the export-time decode (DFL
projection + anchor decode + cls sigmoid) is inherited -> ONNX output is
(1, 4+nc, N) = (1, 5, N) with cx, cy, w, h in letterbox pixel coords.

That is EXACTLY the layout birdcpp.c expects (see line ~1914: layout
(1, 4+nc, N) or transposed; cx,cy,w,h used directly; NMS in C). No custom
export or decoder work needed on the board side.

Channel plan v1 (all multiples of 8/16, NEON-friendly)
------------------------------------------------------
backbone: 16 -> 32 -> 64 (P3/8) -> 96 (P4/16) -> 128 (P5/32, SPPF)
neck:     PAN, out 64 / 96 / 128
head:     BirdDetect - box hidden 32, cls hidden 32 (v8n uses 16/64 -> the
          cls branch is fat even for nc=1; we slim both to 32)
strides:  8/16/32 -> N = 4116 anchors at 448 input (birdcpp cap is 8400)

Usage on the Colab VM
---------------------
    import birdnet
    birdnet.register()          # BEFORE building the model
    from ultralytics import YOLO
    model = YOLO("/content/birdnet_v1.yaml")
    model.train(data="/content/bird_data.yaml", imgsz=448, ...)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Same as ultralytics.utils.torch_utils.autopad."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class BConv(nn.Module):
    """conv-bn-ReLU drop-in for ultralytics Conv (which is conv-bn-SiLU)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BBottleneck(nn.Module):
    """Standard bottleneck with BConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = BConv(c1, c_, k[0], 1)
        self.cv2 = BConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BC2f(nn.Module):
    """CSP bottleneck (C2f layout) built from BConv/BBottleneck."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = BConv(c1, 2 * self.c, 1)
        self.cv2 = BConv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            BBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class BirdSPPF(nn.Module):
    """SPPF with BConv."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = BConv(c1, c_, 1)
        self.cv2 = BConv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class BirdDetect(_make_detect_base()):
    """Detect with slim branches sized for a 1-class detector.

    ultralytics Detect.__init__ hardcodes:
        c2 = max(16, ch[0] // 4, 16)      # box branch hidden
        c3 = max(ch[0], min(nc, 100))     # cls branch hidden -> 64 for v8n
    For nc=1 the cls branch is wildly oversized (64 hidden convs to produce
    ONE channel). We rebuild both branches at 32 hidden channels.
    Everything else (DFL decode, anchors, sigmoid, export path) inherited.
    """

    def __init__(self, nc=1, ch=()):
        super().__init__(nc, ch)
        c2 = 32  # box branch hidden
        c3 = 32  # cls branch hidden
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                BConv(x, c2, 3),
                BConv(c2, c2, 3),
                nn.Conv2d(c2, self.reg_max * 4, 1),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(BConv(x, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )


def _make_detect_base():
    """Import Detect lazily so the module can be imported without ultralytics."""
    from ultralytics.nn.modules.head import Detect

    return Detect


# ---------------------------------------------------------------- register --
def register():
    """Monkey-patch ultralytics.nn.tasks so standard yaml names resolve to
    our ReLU variants. Call once, BEFORE constructing the model."""
    import ultralytics.nn.tasks as ultasks

    ultasks.Conv = BConv
    ultasks.C2f = BC2f
    ultasks.SPPF = BirdSPPF
    ultasks.Detect = BirdDetect
    # also alias the helper names some parse_model versions look up
    for name, cls in (("BBottleneck", BBottleneck), ("BConv", BConv), ("BC2f", BC2f)):
        setattr(ultasks, name, cls)


def build(yaml_path: str, verbose: bool = True):
    """Register modules + build the DetectionModel from a yaml file."""
    register()
    from ultralytics import YOLO

    return YOLO(yaml_path, verbose=verbose)
