#!/usr/bin/env python3
# board_camera_server.py - ALL-ON-BOARD bird pipeline for TI SK-AM62
#
#   camera source  (phone DroidCam MJPEG, or CSI camera /dev/videoN [IMX219])
#        |  raw frames read by this process
#        |  CSI path uses the camkit backend chain:
#           frame_pumper.c pipe (4-thread parallel DMA copy, ~15 fps)
#             -> ctypes MMAP (DQBUF) -> plain read (last resort)
#        v  fresh frames -> board_detection.run() IN-PROCESS (cached ONNX net)
#        |  boxes drawn on the board, served from RAM
#        v  this server on :8888 hosts: dashboard, raw frames, det image,
#           metadata JSON, camera status (auto-reconnect every 2 s shown live)
#
# NOTHING is ever written to disk while the live feed runs (no meta.json,
# no live_det.jpg, no numbered snapshots) - results live in RAM.
#
# Efficiency notes: in-process inference (no subprocess per frame), ONNX net
# cached at first load, frames processed once, light dashboard that stops
# polling while hidden / camera-down, frame-stall watchdog -> reconnect.
#
# Efficiency notes: in-process inference (no subprocess per frame), ONNX net
# cached at first load, frames processed once, light dashboard that stops
# polling while hidden / camera-down, frame-stall watchdog -> reconnect.
#
# Camera source is read from /home/root/bird/camera_url.txt each reconnect
# attempt, so you can change it without restarting this service.
# Write "/dev/video0" in it to use the CSI camera; an http:// URL uses DroidCam.

import os, sys, time, json, threading, http.client, subprocess, select, fcntl, re
from urllib.parse import parse_qs, urlsplit
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BIRD   = "/home/root/bird"
URLFILE= os.path.join(BIRD, "camera_url.txt")
# Objects that used to be written to disk (frame.jpg / live_det.jpg /
# meta.json / numbered live_*_det.jpg snapshots) are gone: detection
# results and the annotated image live in process memory and are served
# straight from RAM. The board writes NOTHING while the live feed runs.

RETRY_EVERY     = 2.0     # seconds between camera reconnect attempts
CONNECT_TIMEOUT = 8       # seconds for the first HTTP connect
STREAM_TIMEOUT  = 20      # seconds with no bytes -> reconnect
FRAME_STALL     = 15      # seconds with NO complete jpeg frame -> reconnect
                          # (the phone can keep the socket alive but stop
                          #  streaming, e.g. DroidCam when the screen locks)

# user-tuned defaults (sensor 1640x1232 2x2 binned, ~27.6 fps at vblank 683)
_DEF_WANT = {"exposure": 1759, "analogue_gain": 200, "digital_gain": 4095,
             "vertical_blanking": 683, "test_pattern": 0,
             "red_pixel_value": 1023, "green_red_pixel_value": 1023,
             "green_blue_pixel_value": 1023, "blue_pixel_value": 1023,
             "horizontal_flip": 0, "vertical_flip": 0}

lock = threading.Lock()
S = {
    "jpeg": None, "jpeg_t": 0.0, "frames": 0,
    "det_jpg": None,
    "cam_wb": 1, "cam_gamma": 1,   # dashboard color toggles (CSI view path)
    "mode": "1640x1232", "scale": "half", "quality": 80,
    "vw": 820, "vh": 616,          # current view (preview) pixel dims
    "want": dict(_DEF_WANT),       # desired sensor + orientation state
    "ctl": {}, "ranges": {},       # kernel read-back + driver ranges
    "line_us": None,               # measured line time us (fps preset math)
    "reconf": False,               # capture loop must tear down + rebuild
    "meta_txt": b'{"n":null,"infer_ms":null,"err":"warming up"}',
    "last_infer_t": 0.0,
    "cam": dict(url="", connected=False, attempts=0, note="starting",
                last_try=0.0, last_ok=0.0, backend="?", fps=0.0),
}

def set_cam(**kw):
    with lock:
        S["cam"].update(kw)
        if "last_try" not in kw:
            S["cam"]["last_try"] = time.time()

BD = None   # board_detection module, imported lazily in the infer thread

def set_meta_error(msg):
    body = json.dumps({"n": None, "infer_ms": None, "err": msg}).encode()
    with lock:
        S["meta_txt"] = body

# ---------------- CSI camera (/dev/videoN) support ----------------
# Verified empirically on this board (Arducam IMX219 -> merged DTB -> csi2rx):
#   - pad wiring: imx219 4-0010 -> cdns csi-bridge -> ticsi2rx (/usr/bin/media-ctl)
#   - capture:    /dev/video0 'RG10' 1640x1232 (2x2 binned), copied by the
#                 frame_pumper C helper (see the backends below)
#   - color phase: cv2.COLOR_BayerBG2BGR  (proven with the sensor's solid-red
#                 test pattern: gives R=255,B=0,G=0; every other phase ~1.0)
#   - brightness: user-tuned analogue_gain 200, digital_gain 4095,
#                 exposure 1759 lines, vblank 683 (~27.6 fps)
_np = None
_cv = None

def _ensure_cv():
    """Import cv2/numpy only when the CSI camera is first used, so a
    DroidCam-only install keeps its original boot cost."""
    global _np, _cv
    if _cv is None:
        import numpy as npv
        import cv2 as cvv
        _np, _cv = npv, cvv
    return _np, _cv

CSI_SENSOR_DEV = "/dev/v4l-subdev2"     # imx219 subdevice node
V4L            = "/usr/bin/v4l2-ctl"
MEDIA          = "/usr/bin/media-ctl"
MEDIA_DEV      = "/dev/media0"

# sensor modes (all SRGGB10 Bayer, 2 bytes/pixel):
MODES  = {"1640x1232": (1640, 1232), "1920x1080": (1920, 1080),
          "3280x2464": (3280, 2464)}
SCALES = {"half": 0.5, "34": 0.75, "full": 1.0}
SCALE_LABEL = {"half": "half", "34": "3/4", "full": "full"}

SCALARS    = ["exposure", "analogue_gain", "digital_gain", "vertical_blanking",
              "test_pattern", "red_pixel_value", "green_red_pixel_value",
              "green_blue_pixel_value", "blue_pixel_value"]
FMT_FIELDS = {"exposure": "exposure", "analogue_gain": "gain",
              "digital_gain": "dig", "vertical_blanking": "vblank",
              "test_pattern": "pattern", "red_pixel_value": "pr",
              "green_red_pixel_value": "pgr", "green_blue_pixel_value": "pgb",
              "blue_pixel_value": "pb"}
RANGEB     = {"exposure": (4, 1911), "analogue_gain": (0, 232),
              "digital_gain": (256, 4095), "vertical_blanking": (4, 64303),
              "test_pattern": (0, 4), "red_pixel_value": (0, 1023),
              "green_red_pixel_value": (0, 1023),
              "green_blue_pixel_value": (0, 1023), "blue_pixel_value": (0, 1023)}

def csi_configure(dev, W, H):
    """Propagate the wanted sensor mode through the whole capture chain
    (subdev -> csi2rx -> ticsi2rx -> video node). Subprocesses fire once
    per (re)open, never per frame. Sensor scalar controls and flips are
    applied by apply_want() right after this - flips are only honoured
    while the device is NOT streaming, so apply_want runs pre-STREAMON."""
    cmds = [
        [MEDIA, "-d", MEDIA_DEV, "-V",
         '"imx219 4-0010":0 [fmt:SRGGB10_1X10/%dx%d field:none]' % (W, H)],
        [MEDIA, "-d", MEDIA_DEV, "-V",
         '"cdns_csi2rx.30101000.csi-bridge":0 [fmt:SRGGB10_1X10/%dx%d field:none]' % (W, H)],
        [MEDIA, "-d", MEDIA_DEV, "-V",
         '"30102000.ticsi2rx":0 [fmt:SRGGB10_1X10/%dx%d field:none]' % (W, H)],
        [V4L, "-d", dev,
         "--set-fmt-video=width=%d,height=%d,pixelformat=RG10" % (W, H)],
    ]
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=15)
        except Exception:
            raise RuntimeError("csi config tool missing: %s" % cmd[0])
        if r.returncode != 0:
            out = (r.stdout or b"").decode(errors="replace").strip()[-100:]
            raise RuntimeError("csi config failed: %s ... %s"
                               % (" ".join(cmd[:6]), out))


def run_tools(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return r.returncode, (r.stdout or b"").decode(errors="replace")
    except Exception as e:
        return 99, repr(e)


def get_ctl():
    """Read current values + ranges for every control from the kernel."""
    rc, out = run_tools([V4L, "-d", CSI_SENSOR_DEV, "-l"], 10)
    if rc:
        return {}, {}
    vals, rngs = {}, {}
    for line in out.splitlines():
        m = re.match(r"\s*([a-zA-Z_0-9]+)\s+0x[0-9a-f]+\s+\(\S+\)\s*:", line)
        if not m:
            continue
        name = m.group(1)
        mm = re.search(r"min=(-?\d+)\s+max=(-?\d+)", line)
        vm = re.search(r"(?:^|\s)value=(-?\d+)", line)
        if mm:
            rngs[name] = (int(mm.group(1)), int(mm.group(2)))
        if vm:
            vals[name] = int(vm.group(1))
    return vals, rngs


def sensor_set(kv):
    """Apply sensor controls - one combined v4l2-ctl call (each -c its own
    arg; the comma-separated form is rejected by this driver)."""
    args = []
    for k, v in kv.items():
        args += ["-c", "%s=%d" % (k, int(v))]
    return run_tools([V4L, "-d", CSI_SENSOR_DEV] + args)


def clamp(name, v, merged):
    lo, hi = merged.get(name, RANGEB.get(name, (0, 0)))
    return max(int(lo), min(int(hi), int(v)))


def view_dims():
    """View dims for the current mode + scale (even numbers)."""
    f = SCALES.get(S["scale"], 0.5)
    W, H = MODES[S["mode"]]
    return (max(2, (int(W * f) // 2) * 2), max(2, (int(H * f) // 2) * 2))


def apply_want(WH):
    """Clamp every desired control to the driver's current ranges and push
    it with the flips in ONE call. Call while the video node is open but
    BEFORE STREAMON (flip requirement). Also seeds S/line_us from the
    kernel-reported geometry for the fps preset."""
    with lock:
        want = dict(S["want"])
    vals, ranges = get_ctl()
    merged = dict(RANGEB)
    merged.update(ranges)
    kv = {}
    for k in SCALARS:
        kv[k] = clamp(k, want.get(k, _DEF_WANT[k]), merged)
    kv["horizontal_flip"] = int(want.get("horizontal_flip", 0))
    kv["vertical_flip"] = int(want.get("vertical_flip", 0))
    rc, out = sensor_set(kv)
    if rc:
        raise RuntimeError("sensor apply rejected: %s" % out.strip()[-120:])
    ctl = get_ctl()[0]
    hb, pr = ctl.get("horizontal_blanking"), ctl.get("pixel_rate")
    with lock:
        S["want"] = dict(want)
        for k in SCALARS:
            S["want"][k] = ctl.get(k, kv[k])
        S["want"]["horizontal_flip"] = kv["horizontal_flip"]
        S["want"]["vertical_flip"] = kv["vertical_flip"]
        S["ranges"] = ranges
        S["ctl"] = ctl
        if hb and pr:
            S["line_us"] = (WH[0] + hb) / float(pr) * 1e6
    with lock:
        S["vw"], S["vh"] = view_dims()

# ---- capture backends (ported from the camkit lab, see CAMERA_PIPELINE.md) ----
# The plain os.read path used before leaves ~11 fps on the table: one 4 MB
# DMA frame must be copied out of UNCACHED memory per read here. The proven
# chain is:  pipe (frame_pumper.c, 4-thread parallel copy, ~15 fps)
#         -> ctypes mmap (DQBUF, zero queue ops per frame)
#         -> plain read (last resort).
import ctypes
import mmap as _mmmod

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
libc.ioctl.restype = ctypes.c_int

_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_V4L2_MEMORY_MMAP = 1


class _Timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class _BufM(ctypes.Union):
    _fields_ = [("offset", ctypes.c_uint32), ("userptr", ctypes.c_ulong),
                ("fd", ctypes.c_int32)]


class v4l2_buffer(ctypes.Structure):
    class _Ts(ctypes.Structure):
        _fields_ = [("sec", ctypes.c_long), ("usec", ctypes.c_long)]
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("field", ctypes.c_uint32),
                ("timestamp", _Ts), ("timecode", _Timecode),
                ("sequence", ctypes.c_uint32), ("memory", ctypes.c_uint32),
                ("m", _BufM), ("length", ctypes.c_uint32),
                ("reserved2", ctypes.c_uint32), ("request_fd", ctypes.c_int32)]


class v4l2_requestbuffers(ctypes.Structure):
    # Board kernel (5.10 vendor tree) ABI proven at 20 bytes:
    # capabilities u32 + reserved[4].  (Mainline 5.10.x headers carry a
    # 24-byte / reserved[8] variant - do not "fix" this without a probe.)
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("capabilities", ctypes.c_uint32),
                ("reserved", ctypes.c_uint8 * 4)]


def _iowr(nr, sz):
    return (3 << 30) | (sz << 16) | (0x56 << 8) | nr


def _iow_(nr, sz):
    return (1 << 30) | (sz << 16) | (0x56 << 8) | nr


_VB_SZ = ctypes.sizeof(v4l2_buffer)
_VRB_SZ = ctypes.sizeof(v4l2_requestbuffers)
IO_REQBUFS = _iowr(8, _VRB_SZ)
IO_QUERYBUF = _iowr(9, _VB_SZ)
IO_QBUF = _iowr(15, _VB_SZ)
IO_DQBUF = _iowr(17, _VB_SZ)
IO_STREAMON = _iow_(18, 4)
IO_STREAMOFF = _iow_(19, 4)


class MMapCap:
    """V4L2 MMAP streaming backend. DQBUF per frame; DMA memory reads are
    UNCACHED on this platform (~12 ms per 4 MB frame), so one bytes()
    memcpy into cache is still the fastest read."""
    name = "mmap"

    def __init__(self, fd, npix, W, H, dev):
        try:
            self._init(fd, npix)
        except Exception:
            # never leave REQBUFS/STREAMON state behind: the vb2 queue is
            # shared on this driver, leaked buffers make every later
            # S_FMT fail with EBUSY forever
            self.stop()
            raise

    def _init(self, fd, npix):
        if _VB_SZ != 88:
            raise RuntimeError("v4l2_buffer ABI size %d != 88" % _VB_SZ)
        if IO_DQBUF != 0xC0585611:
            raise RuntimeError("DQBUF ioctl number %x != C0585611" % IO_DQBUF)
        self.fd = fd
        self.npix = npix
        self.reqsz = 0
        rb = v4l2_requestbuffers()
        rb.count, rb.type, rb.memory = 4, _V4L2_BUF_TYPE_VIDEO_CAPTURE, _V4L2_MEMORY_MMAP
        ok = False
        for sz in (_VRB_SZ, 24, 16, 20, 28, 36, 40, 44):
            ctypes.set_errno(0)
            if libc.ioctl(fd, _iowr(8, sz), ctypes.byref(rb)) == 0:
                ok = True
                self.reqsz = sz
                break
        if not ok:
            raise RuntimeError("REQBUFS rejected at all known ABI sizes")
        self.bufs = []
        for i in range(rb.count):
            b = v4l2_buffer()
            b.index, b.type, b.memory = i, _V4L2_BUF_TYPE_VIDEO_CAPTURE, _V4L2_MEMORY_MMAP
            if libc.ioctl(fd, IO_QUERYBUF, ctypes.byref(b)) != 0:
                raise RuntimeError("QUERYBUF errno %d" % ctypes.get_errno())
            length = int(b.length)
            if length < npix * 2:
                raise RuntimeError("mmap buffer %d < frame %d" % (length, npix * 2))
            mm = _mmmod.mmap(fd, length, access=_mmmod.ACCESS_READ,
                             offset=int(b.m.offset))
            self.bufs.append((mm, length))
        for i in range(len(self.bufs)):
            self._q(i)
        on = ctypes.c_int(1)
        if libc.ioctl(fd, IO_STREAMON, ctypes.byref(on)) != 0:
            raise RuntimeError("STREAMON errno %d" % ctypes.get_errno())
        self.on = True

    def _q(self, idx):
        b = v4l2_buffer()
        b.index = idx
        b.type = _V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = _V4L2_MEMORY_MMAP
        if libc.ioctl(self.fd, IO_QBUF, ctypes.byref(b)) != 0:
            raise RuntimeError("QBUF errno %d" % ctypes.get_errno())

    def fetch(self):
        """Returns (uint16 array view, buffer index). The array is only
        valid until put(idx) requeues the buffer."""
        b = v4l2_buffer()
        b.type = _V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = _V4L2_MEMORY_MMAP
        if libc.ioctl(self.fd, IO_DQBUF, ctypes.byref(b)) != 0:
            raise BlockingIOError("DQBUF errno %d" % ctypes.get_errno())
        mm, length = self.bufs[b.index]
        n = min(int(b.bytesused), length)
        if n < self.npix * 2:
            raise RuntimeError("short frame (%d/%d)" % (n, self.npix * 2))
        data = bytes(mm[:self.npix * 2])
        return _np.frombuffer(data, "<u2"), int(b.index)

    def put(self, idx):
        try:
            self._q(idx)
        except RuntimeError:
            pass

    def stop(self):
        if getattr(self, "on", False):
            off = ctypes.c_int(1)
            libc.ioctl(self.fd, IO_STREAMOFF, ctypes.byref(off))
            self.on = False
        for mm, _ in getattr(self, "bufs", []):
            try:
                mm.close()
            except Exception:
                pass
        self.bufs = []
        try:
            rb = v4l2_requestbuffers()
            rb.count, rb.type, rb.memory = 0, _V4L2_BUF_TYPE_VIDEO_CAPTURE, _V4L2_MEMORY_MMAP
            libc.ioctl(self.fd, IO_REQBUFS, ctypes.byref(rb))
        except Exception:
            pass


def _frame_pumper_path():
    for p in ("/home/root/bird/frame_pumper",
              "/home/root/camkit/frame_pumper"):
        if os.path.exists(p):
            return p
    return None


class PipeCap:
    """frame_pumper.c child - the fast backend. The C helper does the
    V4L2 mmap stream and copies each 4 MB frame to stdout, splitting the
    copy across 4 threads (uncached DMA reads are latency-bound, so
    parallel streams multiply effective bandwidth ~1.5x). This side reads
    the pipe with os.readv straight into a persistent frame buffer - no
    python-side per-frame copy at all."""
    name = "pipe"

    def __init__(self, fd, npix, W, H, dev):
        exe = _frame_pumper_path()
        if exe is None:
            raise RuntimeError("frame_pumper binary not deployed")
        self.fd = fd                      # format owner stays open
        self.npix = npix
        self.fb = bytearray(npix * 2)
        self.fbv = memoryview(self.fb)
        self.pos = 0
        # PR_SET_PDEATHSIG inside the child: it dies with us, never
        # leaks the camera across service restarts
        self.p = subprocess.Popen([exe, dev, str(W), str(H), "4"],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        self.rfd = self.p.stdout.fileno()
        os.set_blocking(self.rfd, False)
        # 4 MB frames squeeze through the default 64 KB pipe in tiny
        # gulps - widen it to 1 MB (F_SETPIPE_SZ)
        try:
            fcntl.fcntl(self.rfd, 1031, 1 << 20)
        except Exception:
            pass

    def fetch(self):
        """Fill the persistent buffer straight from the pipe (os.readv =
        kernel writes into cached userspace). Raises BlockingIOError
        while a frame is still in flight; progress persists in .pos."""
        need = self.npix * 2
        pos = self.pos
        while pos < need:
            n = os.readv(self.rfd, [self.fbv[pos:]])
            if n <= 0:   # EAGAIN raises BlockingIOError instead; 0 = EOF
                raise RuntimeError("pumper stream ended")
            pos += n
        self.pos = 0
        return _np.frombuffer(self.fb, "<u2", count=self.npix), -1

    def put(self, idx):
        pass

    def stop(self):
        try:
            self.p.terminate()
            self.p.wait(timeout=3)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass
        try:
            self.p.stdout.close()
        except Exception:
            pass


class ReadCap:
    """Slowest fallback: vague old plain os.read loop (~11 fps)."""
    name = "read"

    def __init__(self, fd, npix, W, H, dev):
        self.fd = fd
        self.npix = npix

    def fetch(self):
        data = os.read(self.fd, self.npix * 2)
        if len(data) != self.npix * 2:
            raise RuntimeError("short frame (%d/%d)" % (len(data), self.npix * 2))
        return _np.frombuffer(data, "<u2"), -1

    def put(self, idx):
        pass

    def stop(self):
        pass


_WB = [1.0, 1.0, 1.0]       # smoothed per-BGR-channel gains (gray-world AWB)
_LUTSIG = None              # settings signature the composed LUT was built for
_CLUT = None                # (1,256,3) uint8 WB+gamma composed LUT

def _build_color_lut(wb_on, gamma_on):
    """Compose white balance + shadow gamma into ONE uint8 LUT; rebuild
    only when the EMA gains or the dashboard toggles change. One SIMD
    cv2.LUT call per frame beat the per-pixel float math (which ate
    ~10 ms at 720p on the A53)."""
    global _LUTSIG, _CLUT
    sig = (wb_on, gamma_on,
           round(_WB[0], 3), round(_WB[1], 3), round(_WB[2], 3))
    if sig == _LUTSIG and _CLUT is not None:
        return _CLUT
    lut = _np.empty((1, 256, 3), _np.uint8)
    for c in range(3):
        g = _WB[c] if wb_on else 1.0
        col = []
        for i in range(256):
            v = i
            if gamma_on:
                v = (i / 255.0) ** 0.55 * 255.0
            col.append(min(255, int(v * g + 0.5)))
        lut[0, :, c] = col
    _CLUT = lut
    _LUTSIG = sig
    return lut

BA4 = ("BG", "GB", "RG", "GR")   # Bayer phases; OpenCV code resolves lazily

def bayer_code(phase):
    """cv2.COLOR_Bayer<phase>2BGR (cv2 is imported lazily at first use)."""
    return getattr(_cv, "COLOR_Bayer" + (phase if phase in BA4 else "BG") + "2BGR")

def csi_produce_jpg(raw_u16, W, H, phase):
    """Raw frame -> 8-bit -> BGR (phase verified per session) -> view
    dims -> white balance -> shadow gamma -> JPEG."""
    np, cv = _np, _cv
    with lock:
        pw, ph = S["vw"], S["vh"]
        wb_on = int(S["cam_wb"]) != 0
        gam_on = int(S["cam_gamma"]) != 0
        q = int(S["quality"])
    b8 = (raw_u16.reshape(H, W) >> 2).astype(np.uint8)
    bgr = cv.cvtColor(b8, bayer_code(phase))
    if (pw, ph) != (W, H):
        bgr = cv.resize(bgr, (pw, ph), interpolation=cv.INTER_AREA)
    if wb_on:
        # white balance: gray-world on a sparse sample, EMA-smoothed
        sub = bgr[::16, ::16].astype(np.float32)
        m = sub.mean(axis=(0, 1))                # B,G,R means
        if m[1] > 8.0 and m[0] > 4.0 and m[2] > 4.0:
            target = np.clip(m[1] / np.maximum(m, 1.0), 0.6, 2.2)
            _WB[0] = 0.95 * _WB[0] + 0.05 * float(target[0])
            _WB[1] = 0.95 * _WB[1] + 0.05 * float(target[1])
            _WB[2] = 0.95 * _WB[2] + 0.05 * float(target[2])
    # per-channel WB + shadow gamma (dim-room visibility) in one LUT pass
    if wb_on or gam_on:   # both off => untouched source pixels
        bgr = cv.LUT(bgr, _build_color_lut(wb_on, gam_on))
    ok, jpg = cv.imencode(".jpg", bgr,
                          [cv.IMWRITE_JPEG_QUALITY, max(30, min(95, q))])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return jpg.tobytes()


def _fetch_timeout(cap, rfd, tmo=2.0):
    """One frame with a select wall-clock; None on timeout / EAGAIN."""
    if not select.select([rfd], [], [], tmo)[0]:
        return None
    try:
        return cap.fetch()
    except BlockingIOError:
        return None


def verify_phase(cap, rfd, W, H):
    """Re-verify the Bayer phase with the sensor's solid-red test pattern
    (proven recipe: only the correct OpenCV code yields R=255, G=B=0).
    Needed per session because mode changes can swap the phase. Restores
    the user's test-pattern controls afterwards."""
    _ensure_cv()
    np, cv = _np, _cv
    with lock:
        want = dict(S["want"])
    restore = {"test_pattern": int(want.get("test_pattern", 0)),
               "red_pixel_value": int(want.get("red_pixel_value", 1023)),
               "green_red_pixel_value": int(want.get("green_red_pixel_value", 1023)),
               "green_blue_pixel_value": int(want.get("green_blue_pixel_value", 1023)),
               "blue_pixel_value": int(want.get("blue_pixel_value", 1023))}
    best, best_score = "BG", 0.0
    try:
        sensor_set({"test_pattern": 2, "red_pixel_value": 1023,
                    "green_red_pixel_value": 0, "green_blue_pixel_value": 0,
                    "blue_pixel_value": 0})
        got = 0
        while got < 2:
            r = _fetch_timeout(cap, rfd, 2.5)
            if r is None:
                break
            raw, idx = r
            cap.put(idx)
            got += 1
            b8 = (raw.reshape(H, W) >> 2).astype(np.uint8)
            for name in BA4:
                bgr = cv.cvtColor(b8, getattr(cv, "COLOR_Bayer%s2BGR" % name))
                rmean = float(bgr[..., 2].mean())
                bmean = float(bgr[..., 0].mean())
                gmean = float(bgr[..., 1].mean())
                score = rmean / max(0.5 * (bmean + gmean), 1.0)
                if score > best_score:
                    best, best_score = name, score
    finally:
        sensor_set(restore)
    return best


class Reconfigure(Exception):
    """Raised inside the capture session when the dashboard changed mode
    or flips: csi_session must tear the whole pipeline down and rebuild."""


def csi_session(dev, attempts):
    """CSI capture session for the CURRENT S["mode"]: configure pipeline
    pads + format, apply all sensor controls + flips (pre-STREAMON),
    verify the Bayer phase with the solid-red pattern, then run the
    backend chain (frame_pumper pipe -> ctypes mmap -> plain read).
    Same reconnect semantics as the http camera."""
    set_cam(url=dev, connected=False, attempts=attempts, backend="?",
            note="CSI camera open, wiring pipeline pads...")
    last_ok = 0.0
    fidx = 0
    fps = 0.0
    fd = None
    cap = None
    phase = "BG"
    try:
        with lock:
            mode = S["mode"]
        W, H = MODES[mode]
        csi_configure(dev, W, H)
        _ensure_cv()
        fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)   # RDWR: STREAMON needs it
        # flips + all scalars BEFORE the backend builds (pre-STREAMON)
        set_cam(note="CSI camera (%s): applying sensor settings..." % dev)
        apply_want((W, H))
        errs = []
        backend = "read"
        for bcls in (PipeCap, MMapCap):
            try:
                cap = bcls(fd, W * H, W, H, dev)
                backend = bcls.name
                break
            except Exception as e:
                errs.append("%s:%s" % (bcls.name, str(e)[:60]))
        if cap is None:
            cap = ReadCap(fd, W * H, W, H, dev)
            set_cam(backend="read",
                    note="%s: slow read fallback (%s)" % (dev, " | ".join(errs)))
        else:
            set_cam(backend=backend)
        rfd = getattr(cap, "rfd", fd)
        phase = verify_phase(cap, rfd, W, H)
        while True:
            with lock:
                if S["reconf"]:                       # dashboard mode/flip change
                    S["reconf"] = False
                    raise Reconfigure("settings changed on the dashboard")
            if not select.select([rfd], [], [], 3.0)[0]:
                if (time.time() - last_ok) > FRAME_STALL:
                    raise RuntimeError("no camera frames for %d s - reconnecting"
                                       % FRAME_STALL)
                continue
            try:
                raw, bidx = cap.fetch()
            except BlockingIOError:
                continue
            now = time.time()
            if last_ok:
                d = now - last_ok
                if d > 0 and fidx >= 3:   # fidx<3: ring warmup lies
                    fps = (1.0 / d) if fps <= 0 else fps * 0.9 + (1.0 / d) * 0.1
            last_ok = now
            jpg = csi_produce_jpg(raw, W, H, phase)
            cap.put(bidx)
            with lock:
                S["jpeg"] = jpg
                S["jpeg_t"] = now
                S["frames"] += 1
            fidx += 1
            if fidx == 1 or (fidx % 150 == 0):
                set_cam(connected=True, last_ok=now, attempts=0, fps=fps,
                        note="CSI camera %s: LIVE (%.1f fps, %s backend, "
                             "phase %s)" % (dev, fps, backend, phase))
    except Reconfigure:
        with lock:
            S["note"] = "settings changed - rebuilding camera session..."
        raise
    except Exception as e:
        set_cam(connected=False,
                note="camera not connected - retrying every 2 s (%s)" % str(e)[:60])
    finally:
        if cap is not None:
            try:
                cap.stop()
            except Exception:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass


def _stream_sock(conn, resp):
    """The live stream socket. On DroidCam's multipart response the HTTP
    connection hands ownership of the socket to the response object, so
    conn.sock is None there and the real socket lives in resp.fp.raw._sock."""
    s = getattr(conn, "sock", None)
    if s is not None:
        return s
    try:
        return resp.fp.raw._sock
    except Exception:
        return None

def capture_loop():
    """Pull raw JPEGs from the camera URL, auto-reconnect every 2 s."""
    while True:
        url = ""
        try:
            url = open(URLFILE).read().strip()
        except Exception:
            pass
        attempts = S["cam"]["attempts"] + 1
        if not url:
            set_cam(connected=False, url="", attempts=attempts,
                    note="no camera source configured - retrying every 2 s")
            time.sleep(RETRY_EVERY)
            continue
        if url.startswith("/dev/"):
            try:
                csi_session(url, attempts)
            except Reconfigure:
                continue          # settings changed: rebuild instantly
            time.sleep(RETRY_EVERY)
            continue
        set_cam(url=url, connected=False, attempts=attempts,
                note="connect to camera...")
        got_frame = False
        scanned = 0
        last_frame_t = time.time()
        conn = None
        try:
            parts = urlsplit(url)
            host = parts.hostname or "0.0.0.0"
            port = parts.port or 80
            path = parts.path or "/"
            conn = http.client.HTTPConnection(host, port, timeout=CONNECT_TIMEOUT)
            conn.request("GET", path, headers={"User-Agent": "board-bird/1.0"})
            resp = conn.getresponse()
            # The phone answers with an HTML page instead of video when
            # DroidCam is busy with another client - fail fast so /phone
            # reports "retrying" honestly instead of hanging on a dead body.
            ctype = (resp.getheader("Content-Type") or "").lower()
            if "text/html" in ctype:
                raise RuntimeError("phone busy (DroidCam held by another client)")
            if ctype and not ("multipart" in ctype or "jpeg" in ctype
                              or "octet" in ctype):
                raise RuntimeError("not a video stream (content-type %s)" % ctype[:36])
            try:
                conn.sock.settimeout(STREAM_TIMEOUT)
            except Exception:
                pass
            try:
                resp.fp.raw._sock.settimeout(STREAM_TIMEOUT)
            except Exception:
                pass
            if _stream_sock(conn, resp) is None:
                set_cam(note="warning: cannot set stream socket timeout")
            set_cam(note="reading stream...")
            buf = b""
            while True:
                if (time.time() - last_frame_t) > FRAME_STALL:
                    raise RuntimeError("no camera frames for %d s - reconnecting"
                                       % FRAME_STALL)
                try:
                    chunk = resp.read(65536)
                except Exception:
                    raise RuntimeError("stream ended-timeout - reconnecting")
                if not chunk:
                    raise RuntimeError("stream closed - reconnecting")
                buf += chunk
                scanned += len(chunk)
                if not got_frame and scanned > 262144:
                    raise RuntimeError("no jpeg frames in the stream (not video?)")
                while True:
                    i = buf.find(b"\xff\xd8")
                    if i < 0:
                        buf = buf[-3:]
                        break
                    j = buf.find(b"\xff\xd9", i + 2)
                    if j < 0:
                        buf = buf[i:]
                        break
                    jpg = buf[i:j + 2]
                    buf = buf[j + 2:]
                    last_frame_t = time.time()
                    with lock:
                        S["jpeg"] = jpg
                        S["jpeg_t"] = last_frame_t
                        S["frames"] += 1
                    if not got_frame:
                        got_frame = True
                        set_cam(connected=True, last_ok=time.time(),
                                note="camera: LIVE", attempts=0)
        except Exception as e:
            msg = str(e)
            set_cam(connected=False,
                    note="camera not connected - retrying every 2 s (%s)" % msg[:60])
            # Phone MACs change on reconnect (Android randomized Wi-Fi MAC):
            # a stale ARP entry keeps TCP connects timing out forever. Drop
            # it so the next attempt re-resolves the phone's fresh MAC.
            if "timed out" in msg or "refused" in msg:
                try:
                    host = urlsplit(url).hostname
                    if host:
                        os.system("busybox arp -d %s 2>/dev/null" % host)
                except Exception:
                    pass
        try:
            if conn:
                conn.close()
        except Exception:
            pass
        time.sleep(RETRY_EVERY)

def infer_loop():
    """In-process YOLOv8 on the freshest NEW frame.

    Optimization vs the old per-cycle subprocess:
      - detection runs inside this process (no python restart per cycle,
        no re-import of cv2/numpy, no ONNX model reload - it is cached)
      - a frame is processed once: re-runs on the same frame are skipped
      - JPEG bytes go straight to the decoder in memory (no temp file I/O)
    NOTHING is written to disk: meta stays in memory and the annotated
    image is an in-memory buffer for /det.jpg (the periodic numbered
    live_*_det.jpg snapshots are gone for good)."""
    global BD
    done_t = 0.0
    while True:
        time.sleep(1.2)   # 1/s cadence: the dashboard det panel polls 1.5 s;
                          # each ONNX forward blocks the GIL ~1-2 s, so a
                          # faster cadence only throttles the camera thread
        with lock:
            jpg, jt = S["jpeg"], S["jpeg_t"]
        if jpg is None or (time.time() - jt) > 15 or jt <= done_t:
            continue
        if BD is None:
            try:
                if BIRD not in sys.path:
                    sys.path.insert(0, BIRD)
                import board_detection as bd
                BD = bd
            except Exception as e:
                set_meta_error("detection module failed: %s" % str(e)[:80])
                continue
        done_t = jt               # this frame is handled even if detection errs
        try:
            d = BD.run("best.onnx", "frame.jpg", 640, 0.25, 0.45,
                       verbose=False, jpeg_bytes=jpg, return_annotated=True)
        except Exception as e:
            d = {"n": None, "infer_ms": None,
                 "err": "detection error: %s" % str(e)[:80]}
        body = json.dumps({k: v for k, v in d.items()
                           if k != "annotated_jpg"}).encode()
        ann = d.get("annotated_jpg")
        with lock:
            S["meta_txt"] = body
            S["last_infer_t"] = time.time()
            if ann:
                S["det_jpg"] = ann

_HTML_TPL = b"""<!doctype html><html><head><title>Live Bird Detection (all on TI board)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{background:#0e1116;color:#dbe2ef;font-family:Arial,sans-serif;margin:0;padding:12px;font-size:14px}
h1{font-size:18px;margin:0 0 4px}
.sub{color:#8fa1c7;font-size:12px;margin-bottom:10px}
.grid{display:flex;flex-wrap:wrap;gap:10px}
.card{background:#161b24;border:1px solid #2a3245;border-radius:8px;padding:6px}
.card h3{margin:2px 0 4px;font-size:13px;color:#9ab4e8}
img{max-width:96vw;border-radius:6px;background:#000}
.stats{margin-top:10px;font-size:13px;color:#8fa1c7;line-height:1.8}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#888;margin-right:4px}
.dot.ok{background:#2ecc71}
.dot.warn{background:#e67e22}
.dot.bad{background:#e74c3c}
.warnbox{background:#3a2a18;border:1px solid #7a5a2a;color:#f5c06a;border-radius:8px;padding:6px 10px;margin-top:8px;font-size:12px;display:none}
#ph{color:#8fa1c7;font-size:13px;padding:4px}
a{color:#66aaff}
.cb{display:block;margin:3px 0;font-size:13px}
.hc{color:#6d7fa5;font-size:11px}
input[type="number"],select{background:#0e1116;color:#dbe2ef;border:1px solid #2a3245;border-radius:4px;padding:1px 3px;font-size:12px}
.links{margin:4px 0 0;padding-left:16px;font-size:12px;color:#8fa1c7;line-height:1.7}
button{background:#1c2740;border:1px solid #3d5a99;color:#dfe9ff;border-radius:6px;padding:3px 14px;margin-top:5px;cursor:pointer;font-size:13px}
</style></head><body>
<h1>Live bird detection - running on the TI SK-AM62 board</h1>
<div class="sub">camera &rarr; board capture &rarr; board YOLOv8 &rarr; boxes drawn on board &middot; page served by the board &middot; <a href="/stream">full-screen stream</a></div>
<div class="grid">
 <div class="card"><h3>Camera</h3><img id="cam" alt="camera"></div>
 <div class="card"><h3>Detection (boxes drawn on board)</h3><div id="ph">waiting for first result...</div><img id="det" alt="result"></div>
 <div class="card"><h3>Camera settings</h3>
  <form method="post" action="/camera-set">
   <label class="cb">exposure <input type="number" name="exposure" style="width:74px" value="__EXPOSURE__">
    an. gain <input type="number" name="gain" style="width:54px" value="__GAIN__">
    dig gain <input type="number" name="dig" style="width:60px" value="__DIG__"></label>
   <label class="cb">vblank <input type="number" name="vblank" style="width:66px" value="__VBLANK__">
    fps preset <input type="number" name="fpreset" style="width:50px" placeholder="keep">
    <span class="hc">(blank = keep current)</span></label>
   <label class="cb">test pattern <select name="pattern">__POPT__</select></label>
   <label class="cb">pixels R <input type="number" name="pr" style="width:64px" value="__PR__">
    gA <input type="number" name="pgr" style="width:64px" value="__PGR__">
    gB <input type="number" name="pgb" style="width:64px" value="__PGB__">
    bl <input type="number" name="pb" style="width:64px" value="__PB__">
    <span class="hc">(solid-color pixels, pats 2/3)</span></label>
   <label class="cb">sensor mode <select name="mode">__MOPT__</select>
    view <select name="scale">__SOPT__</select>
    JPEG q <input type="number" name="quality" style="width:50px" value="__QUALITY__"></label>
   <label class="cb"><input type="checkbox" name="hflip" value="1"__HFC__> horizontal flip
    <input type="checkbox" name="vflip" value="1"__VFC__> vertical flip
    <span class="hc">(flips + mode rebuild the pipeline)</span></label>
   <label class="cb"><input type="checkbox" name="wb" value="1"__WBC__> white balance
    <input type="checkbox" name="gamma" value="1"__GMC__> shadow lift</label>
   <button type="submit">apply camera settings</button>
  </form>
 </div>
 <div class="card"><h3>Pages on this server</h3>
  <ul class="links">
   <li><a href="/stream">/stream</a> - full-screen MJPEG live feed</li>
   <li><a href="/frame.jpg">/frame.jpg</a> - latest raw camera frame</li>
   <li><a href="/det.jpg">/det.jpg</a> - latest boxes-drawn result</li>
   <li><a href="/meta.json">/meta.json</a> - detection JSON</li>
   <li><a href="/phone">/phone</a> - camera status JSON</li>
   <li><a id="lab" href="#">camera tuning lab (port 8889)</a> - separate service</li>
  </ul>
 </div>
</div>
<div class="stats">
 <span id="pdot" class="dot"></span><span id="phone">camera: checking...</span> &nbsp;|&nbsp;
 birds: <b id="n">-</b> &nbsp;|&nbsp; infer: <b id="ms">-</b> ms &nbsp;|&nbsp; updated: <b id="t">-</b>
</div>
<div class="stats" id="setstat">settings: -</div>
<div class="warnbox" id="warn"></div>
<script>
var cam=document.getElementById('cam'),det=document.getElementById('det'),ph=document.getElementById('ph');
var pdot=document.getElementById('pdot'),pt=document.getElementById('phone'),warn=document.getElementById('warn');
document.getElementById('lab').href = location.protocol + '//' + location.hostname + ':8889/';
var camLive=false;   // drives image refresh ONLY while frames actually flow
cam.onerror=function(){};
det.onerror=function(){};
det.onload=function(){ph.style.display='none';};
/* camera panel: 150 ms poll (~6-7 fps feel, matches real capture fps; the
   full-screen /stream pushes at capture rate if you want more motion) */
setInterval(function(){ if(camLive&&!document.hidden){cam.src='/frame.jpg?'+Date.now();} },150);
/* detection panel: refresh only when inference can be updating */
setInterval(function(){
  if(!camLive||document.hidden) return;
  var p=new Image(); p.onload=function(){det.src=p.src;}; p.onerror=function(){};
  p.src='/det.jpg?'+Date.now();
},1500);
setInterval(function(){
  if(document.hidden) return;
  fetch('/meta.json',{cache:'no-store'}).then(function(r){return r.json();}).then(function(j){
    document.getElementById('n').textContent=(j.n==null?'-':j.n);
    document.getElementById('ms').textContent=(j.infer_ms==null?'-':Math.round(j.infer_ms));
    document.getElementById('t').textContent=new Date().toLocaleTimeString();
  }).catch(function(){});
},2000);
setInterval(function(){
  if(document.hidden) return;
  fetch('/phone',{cache:'no-store'}).then(function(r){return r.json();}).then(function(j){
    camLive=j.connected;
    if(j.connected){
      pdot.className='dot ok';
      pt.textContent='camera: LIVE - '+(j.frame_age_ms/1000).toFixed(1)+' s old (through the board)';
      warn.style.display='none';
    } else {
      pdot.className='dot warn';
      pt.textContent='camera: not connected - retrying every '+j.retry_s+' s';
      warn.style.display='block';
      warn.textContent=j.note;
    }
    var st=document.getElementById('setstat');
    st.textContent='settings: mode '+j.mode+' | view '+j.preview+'@q'+j.quality+
      ' | driver exp '+(j.ctl.exposure!=null?j.ctl.exposure:j.want.exposure)+
      ' gain '+(j.ctl.analogue_gain!=null?j.ctl.analogue_gain:j.want.analogue_gain)+
      ' dig '+(j.ctl.digital_gain!=null?j.ctl.digital_gain:j.want.digital_gain)+
      ' vblank '+(j.ctl.vertical_blanking!=null?j.ctl.vertical_blanking:j.want.vertical_blanking)+
      (j.line_us!=null?(' | line '+j.line_us+' us'):'');
  }).catch(function(){});
},2000);
</script></body></html>"""

def page_html():
    """Dashboard HTML with every camera setting's current value baked in."""
    with lock:
        want = dict(S["want"])
        mode, scale, quality = S["mode"], S["scale"], S["quality"]
        wb, gm = int(S["cam_wb"]), int(S["cam_gamma"])

    def sel(cur, val):
        return ' selected' if cur == val else ''

    pat = int(want.get("test_pattern", 0))
    popt = "".join('<option value="%d"%s>%s</option>' % (
        v, sel(pat, v),
        {0: "0 - off", 2: "2 - solid color"}.get(v, "%d" % v))
        for v in range(5)).encode()
    mopt = "".join('<option value="%s"%s>%s</option>' % (m, sel(mode, m), m)
                   for m in MODES).encode()
    sopt = "".join('<option value="%s"%s>%s</option>'
                   % (s, sel(scale, s), SCALE_LABEL[s])
                   for s in ("half", "34", "full")).encode()
    return (_HTML_TPL
            .replace(b"__EXPOSURE__", b"%d" % int(want.get("exposure", 1759)))
            .replace(b"__GAIN__", b"%d" % int(want.get("analogue_gain", 200)))
            .replace(b"__DIG__", b"%d" % int(want.get("digital_gain", 4095)))
            .replace(b"__VBLANK__", b"%d" % int(want.get("vertical_blanking", 683)))
            .replace(b"__POPT__", popt)
            .replace(b"__PR__", b"%d" % int(want.get("red_pixel_value", 1023)))
            .replace(b"__PGR__", b"%d" % int(want.get("green_red_pixel_value", 1023)))
            .replace(b"__PGB__", b"%d" % int(want.get("green_blue_pixel_value", 1023)))
            .replace(b"__PB__", b"%d" % int(want.get("blue_pixel_value", 1023)))
            .replace(b"__MOPT__", mopt)
            .replace(b"__SOPT__", sopt)
            .replace(b"__QUALITY__", b"%d" % int(quality))
            .replace(b"__HFC__", b" checked" if int(want.get("horizontal_flip", 0)) else b"")
            .replace(b"__VFC__", b" checked" if int(want.get("vertical_flip", 0)) else b"")
            .replace(b"__WBC__", b" checked" if wb else b"")
            .replace(b"__GMC__", b" checked" if gm else b""))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def emit(self, code, body=b"", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass
    def do_POST(self):
        """Dashboard form: every writable driver control + view settings.
        Checkbox semantics (wb/gamma/flips): absent = off. Scalar fields:
        absent = unchanged. fps preset (if > 0) overrides vblank.
        Mode/flip changes raise S["reconf"] which the capture loop honours
        by tearing the pipeline down and rebuilding (mode also implies
        driver format changes; flips are only applied pre-STREAMON)."""
        if not self.path.startswith("/camera-set"):
            return self.emit(404, b'{"err":"unknown endpoint"}')
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
        except (TypeError, ValueError):
            n = 0
        body = self.rfile.read(n) if n > 0 else b""
        f = parse_qs(body.decode("utf-8", "replace"))

        def iv(k):
            v = (f.get(k) or [""])[0].strip()
            if not v:
                return None
            try:
                return int(float(v))
            except ValueError:
                return None

        def flag(k):                     # checkbox: absent = off
            v = (f.get(k) or [""])[0]
            return 0 if v.strip().lower() in ("", "0", "off", "false", "no") else 1

        need_reconf = False
        notes = []
        with lock:
            want = dict(S["want"])
            old_mode = S["mode"]
            old_h = int(want.get("horizontal_flip", 0))
            old_v = int(want.get("vertical_flip", 0))
            ranges = dict(S["ranges"])

        # 1) sensor scalars (clamped against live ranges)
        kv = {}
        merged = dict(RANGEB)
        merged.update(ranges)
        for cname, field in FMT_FIELDS.items():
            v = iv(field)
            if v is None:
                continue
            kv[cname] = clamp(cname, v, merged)
        # 2) fps preset -> vertical_blanking via measured line time
        fp = iv("fpreset")
        if fp and fp > 0:
            with lock:
                line_us = S["line_us"]
            W, H = MODES[old_mode]
            if line_us:
                vb = int(1e6 / (fp * line_us)) - H
                vb = clamp("vertical_blanking", vb, merged)
                kv["vertical_blanking"] = vb
                notes.append("preset %d fps -> vblank %d" % (fp, vb))
            else:
                notes.append("fps preset needs live line_time first (retry)")
        for k, v in kv.items():
            want[k] = v
        # 3) pure view settings (no sensor involvement)
        sc = (f.get("scale") or [None])[0]
        if sc in SCALES and sc != S["scale"]:
            S["scale"] = sc
            S["vw"], S["vh"] = view_dims()
            notes.append("view %s" % SCALE_LABEL[sc])
        qv = iv("quality")
        if qv is not None:
            S["quality"] = max(30, min(95, qv))
        S["cam_wb"] = flag("wb")
        S["cam_gamma"] = flag("gamma")
        # 4) mode (full pipeline rebuild) + flips (honoured pre-STREAMON)
        mv = (f.get("mode") or [None])[0]
        if mv in MODES and mv != old_mode:
            S["mode"] = mv
            S["vw"], S["vh"] = view_dims()
            need_reconf = True
            notes.append("mode %s" % mv)
        hv, vv = flag("hflip"), flag("vflip")
        if hv != old_h or vv != old_v:
            want["horizontal_flip"] = hv
            want["vertical_flip"] = vv
            need_reconf = True
            notes.append("flips %d/%d" % (hv, vv))
        S["want"] = want
        S["reconf"] = bool(need_reconf)

        # live-apply the scalar changes right now (no rebuild needed)
        if kv and not need_reconf:
            rc, out = sensor_set(kv)
            if rc:
                notes.append("driver rejected: %s" % out.strip()[-90:])
            else:
                S["ctl"] = get_ctl()[0]
                for k in kv:
                    if k in S["ctl"]:
                        S["want"][k] = S["ctl"][k]
                if "vertical_blanking" in kv:
                    hb, pr = (S["ctl"].get("horizontal_blanking"),
                              S["ctl"].get("pixel_rate"))
                    if hb and pr:
                        S["line_us"] = (MODES[S["mode"]][0]
                                        + hb) / float(pr) * 1e6
        if notes:
            set_cam(note="settings: " + "; ".join(notes)[:150])
        else:
            set_cam(note="settings: no changes in form")
        self.send_response(303)                 # back to the dashboard
        self.send_header("Location", "/")
        self.end_headers()
    def do_GET(self):
        if self.path in ("/", "/view", "/index.html"):
            return self.emit(200, page_html(), "text/html; charset=utf-8")
        if self.path.startswith("/frame.jpg"):
            with lock:
                jpg = S["jpeg"]
            if not jpg:
                return self.emit(404, b'{"err":"no frame yet"}')
            return self.emit(200, jpg, "image/jpeg")
        if self.path.startswith("/det.jpg") or self.path.startswith("/live_det.jpg"):
            with lock:
                jpg = S["det_jpg"]
            if not jpg:
                return self.emit(404, b'{"err":"no result yet"}')
            return self.emit(200, jpg, "image/jpeg")
        if self.path in ("/meta.json", "/meta"):
            with lock:
                t = S["meta_txt"]
            return self.emit(200, t)
        if self.path == "/phone":
            with lock:
                c = dict(S["cam"])
                frames = S["frames"]
                jt = S["jpeg_t"]
                wbv, gmv = S["cam_wb"], S["cam_gamma"]
                mode, scale, qual = S["mode"], S["scale"], S["quality"]
                vw, vh = S["vw"], S["vh"]
                lms = S["line_us"]
                want = dict(S["want"])
                ctl = dict(S["ctl"])
            fa = int((time.time() - jt) * 1000) if jt else None
            body = json.dumps({
                "connected": c["connected"], "note": c["note"], "url": c["url"],
                "attempts": c["attempts"], "retry_s": int(RETRY_EVERY),
                "frame_age_ms": fa, "frames": frames,
                "backend": c.get("backend", "-"), "fps": round(c.get("fps", 0.0), 2),
                "wb": wbv, "gamma": gmv,
                "mode": mode, "scale": SCALE_LABEL.get(scale, scale),
                "quality": qual, "preview": [vw, vh],
                "line_us": round(lms, 4) if lms else None,
                "want": want, "ctl": ctl,
            }).encode()
            return self.emit(200, body)
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=jpegframe")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpg = S["jpeg"]
                    if jpg:
                        self.wfile.write(b"--jpegframe\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode()
                                         + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.06)   # near-capture-rate MJPEG (caps ~16 fps)
            except Exception:
                pass
        if self.path == "/ping":
            return self.emit(200, b"board-ok", "text/plain")
        self.emit(404, b"")

if __name__ == "__main__":
    # pin all cores to performance: the A53 governor otherwise parks the
    # cores low and halves the whole capture+inference pipeline
    try:
        for g in sorted(__import__("glob").glob(
                "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")):
            with open(g, "w") as fh:
                fh.write("performance")
    except Exception:
        pass
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=infer_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    print("Board server live on http://0.0.0.0:8888/  (camera: see %s)" % URLFILE)
    srv.serve_forever()
