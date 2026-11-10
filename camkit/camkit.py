#!/usr/bin/env python3
# camkit.py - standalone CSI-camera tuning lab for the TI SK-AM62 (+ Arducam IMX219)
#
#   http://<board>:8889/       lab UI: live view + every control the driver exposes
#   /mjpg                      live MJPEG stream (multipart)
#   /frame.jpg                 latest frame snapshot
#   /status                    JSON: fps, frames, applied controls, phase, note
#   /set                       POST/GET form -> applies controls (redirects to /)
#   /readctl                   re-read control values straight from the kernel
#
# Separation from the bird system:
#   while this service runs it owns the CSI camera (/dev/video0). The
#   systemd unit stops birdboard when camkit starts and starts birdboard
#   again when camkit stops - one camera, one owner, no fights.
#
# Capture uses the proven recipe (same as the bird server):
#   raw RG10 blocks via os.read (cv2.VideoCapture mis-negotiates RG10 on
#   this build), plain sel+read loop, and the debayer phase is RE-VERIFIED
#   at every open using the sensor's solid-red test pattern, so flips and
#   mode switches never break colors silently.
# Frame rate: the driver has no fps control - frame time comes from
#   vertical_blanking. This lab self-measures the line time after each
#   (re)configuration and turns fps presets into vblank values.

import os, re, sys, time, json, threading, subprocess, select, fcntl
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import numpy as np
import cv2
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
    """Zero-copy V4L2 MMAP streaming backend (gets the sensor's real
    30/40 fps; plain read() stalls ~10 fps because of the 4 MB copy)."""
    name = "mmap"

    def __init__(self, fd, npix, W, H):
        try:
            self._init(fd, npix, W, H)
        except Exception:
            # never leave REQBUFS/STREAMON state behind: the vb2 queue is
            # shared on this driver, leaked buffers make every later
            # S_FMT fail with EBUSY forever
            self.stop()
            raise

    def _init(self, fd, npix, W, H):
        if _VB_SZ != 88:
            raise RuntimeError("v4l2_buffer ABI size %d != 88" % _VB_SZ)
        if IO_DQBUF != 0xC0585611:
            raise RuntimeError("DQBUF ioctl number %x != C0585611" % IO_DQBUF)
        self.fd = fd
        self.npix = npix
        self.reqsz = 0
        # REQBUFS struct size varies between kernel trees (20-byte proven
        # here); probe list so this survives small kernel ABI drift.
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
        """Returns (uint16 array view, buffer index). Array is valid only
        until put(idx) is called."""
        b = v4l2_buffer()
        b.type = _V4L2_BUF_TYPE_VIDEO_CAPTURE
        b.memory = _V4L2_MEMORY_MMAP
        if libc.ioctl(self.fd, IO_DQBUF, ctypes.byref(b)) != 0:
            raise BlockingIOError("DQBUF errno %d" % ctypes.get_errno())
        mm, length = self.bufs[b.index]
        n = min(int(b.bytesused), length)
        if n < self.npix * 2:
            raise RuntimeError("short frame (%d/%d)" % (n, self.npix * 2))
        # DMA memory reads UNCACHED on this platform (~12 ms per MB), so
        # one plain memcpy is the fastest way to get pixels into cache.
        data = bytes(mm[:self.npix * 2])
        return np.frombuffer(data, "<u2"), int(b.index)

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
        for mm, _ in self.bufs:
            try:
                mm.close()
            except Exception:
                pass
        self.bufs = []
        # release the driver's DMA buffer pool (last, after all maps closed)
        try:
            rb = v4l2_requestbuffers()
            rb.count, rb.type, rb.memory = 0, _V4L2_BUF_TYPE_VIDEO_CAPTURE, _V4L2_MEMORY_MMAP
            libc.ioctl(self.fd, IO_REQBUFS, ctypes.byref(rb))
        except Exception:
            pass


class PipeCap:
    """Middle tier: outsources mmap streaming to v4l2-ctl and reads raw
    RG10 blocks over a pipe (~30 fps). Used when ctypes mmap init fails
    for any reason - guaranteed-good backend that cost us many hours."""
    name = "pipe"

    def __init__(self, fd, npix, W, H):
        self.fd = fd                      # keep the device open (format owner)
        self.npix = npix
        self.fb = bytearray(npix * 2)   # persistent zero-copy frame buffer
        self.fbv = memoryview(self.fb)
        self.pos = 0
        # frame_pumper.c: 4-thread parallel copy of the DMA frame into a
        # pipe (uni-processor memcpy from uncached DMA memory tops at
        # ~14 fps on this board; 4 cores push it to ~21) - python then
        # reads back at cache speed
        self.p = subprocess.Popen(["/home/root/camkit/frame_pumper",
                                   "/dev/video0", str(W), str(H), "4"],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        self.rfd = self.p.stdout.fileno()
        try:
            os.set_blocking(self.rfd, False)
        except Exception:
            pass
        # 4 MB frames squeeze through the default 64 KB pipe in tiny
        # gulps - widen it to 1 MB (F_SETPIPE_SZ)
        try:
            fcntl.fcntl(self.rfd, 1031, 1 << 20)
        except Exception:
            pass

    def fetch(self):
        """Fill the persistent frame buffer straight from the pipe
        (os.readv = kernel writes into cached userspace, no python-side
        per-frame copy). Raises BlockingIOError when the pipe is
        momentarily dry - progress in self.pos persists across calls."""
        need = self.npix * 2
        pos = self.pos
        while pos < need:
            n = os.readv(self.rfd, [self.fbv[pos:]])
            if n <= 0:   # EAGAIN raises BlockingIOError instead; 0 = EOF
                raise RuntimeError("pumper stream ended")
            pos += n
        self.pos = 0
        return np.frombuffer(self.fb, "<u2", count=self.npix), -1

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
    """Fallback: plain read() path (~10 fps at 1080p - the 4 MB per-frame
    copy is the limit, not the sensor)."""
    name = "read"

    def __init__(self, fd, npix, W, H):
        self.fd = fd
        self.npix = npix

    def fetch(self):
        data = os.read(self.fd, self.npix * 2)
        if len(data) != self.npix * 2:
            raise RuntimeError("short frame (%d/%d)"
                               % (len(data), self.npix * 2))
        return np.frombuffer(data, "<u2"), -1

    def put(self, idx):
        pass

    def stop(self):
        pass


def fetch_timeout(cap, timeout=2.0):
    """select + fetch; None when nothing arrived in time.
    Caller must cap.put(idx) after consuming the returned array."""
    rfd = getattr(cap, "rfd", cap.fd)
    if not select.select([rfd], [], [], timeout)[0]:
        return None
    try:
        return cap.fetch()
    except BlockingIOError:
        return None

PORT       = 8889
SENS       = "/dev/v4l-subdev2"
CAP        = "/dev/video0"
MEDIA      = "/dev/media0"
PIXEL_RATE = 182_400_000

# sensor post-processing modes the imx219 driver offers (mirrored through
# csi2rx -> ticsi2rx). All deliver SRGGB10_1X10 Bayer.
MODES = {
    "1640x1232": (1640, 1232),
    "1920x1080": (1920, 1080),
    "3280x2464": (3280, 2464),
}
MODE_NOTES = {
    "1640x1232": "2x2 binned - highest fps potential (driver mode max ~40)",
    "1920x1080": "2x2 binned 1080p - bird-camera mode (tuned for 30 fps)",
    "3280x2464": "full sensor - max detail, slowest",
}
DEFAULT_MODE = "1640x1232"

SCALES = {"full": 1.0, "34": 0.75, "half": 0.5}

DEBAYER = {
    "BG": cv2.COLOR_BayerBG2BGR,
    "GB": cv2.COLOR_BayerGB2BGR,
    "RG": cv2.COLOR_BayerRG2BGR,
    "GR": cv2.COLOR_BayerGR2BGR,
}

# ------- view-side white balance / gamma (kills the native green cast) -------
# Raw IMX219 without WB leans green (G response is the strongest); the fix is
# a per-BGR-channel gain table bound with the optional shadow gamma into ONE
# uint8 LUT, applied with a single SIMD cv2.LUT call per frame.
WB_G = [1.0, 1.0, 1.0]     # gray-world EMA gains (single writer: capture thread)
_LUT = None                 # (1,256,3) uint8 LUT, None = WB+gamma both off
_LUT_SIG = None             # gains/settings signature _LUT was built for


def build_lut(wb_on, gamma_on):
    """Rebuild the composed LUT when WB gains or settings drift; cheap.
    Only the capture thread calls this - no lock needed."""
    global _LUT, _LUT_SIG
    sig = (wb_on, gamma_on, round(WB_G[0], 3), round(WB_G[1], 3),
           round(WB_G[2], 3))
    if sig == _LUT_SIG and _LUT is not None:
        return _LUT
    lut = np.empty((1, 256, 3), np.uint8)
    for c in range(3):
        g = WB_G[c] if wb_on else 1.0
        col = []
        for i in range(256):
            v = i
            if gamma_on:
                v = (i / 255.0) ** 0.55 * 255.0
            col.append(min(255, int(v * g + 0.5)))
        lut[0, :, c] = col
    _LUT = lut
    _LUT_SIG = sig
    return lut

# values we manage; defaults = user-tuned settings (camkit session,
# sensor mode 1640x1232 2x2 binned, ~27.6 fps at vblank 683)
DEF_WANT = {
    "exposure": 1759, "analogue_gain": 200, "digital_gain": 4095,
    "vertical_blanking": 683, "test_pattern": 0,
    "red_pixel_value": 1023, "green_red_pixel_value": 1023,
    "green_blue_pixel_value": 1023, "blue_pixel_value": 1023,
    "horizontal_flip": 0, "vertical_flip": 0,
}
RANGE_FALLBACK = {
    "exposure": (4, 1759), "analogue_gain": (0, 232),
    "digital_gain": (256, 4095), "vertical_blanking": (4, 64455),
    "test_pattern": (0, 4), "red_pixel_value": (0, 1023),
    "green_red_pixel_value": (0, 1023), "green_blue_pixel_value": (0, 1023),
    "blue_pixel_value": (0, 1023), "horizontal_flip": (0, 1),
    "vertical_flip": (0, 1),
}
SCALARS = ["exposure", "analogue_gain", "digital_gain", "vertical_blanking",
           "test_pattern", "red_pixel_value", "green_red_pixel_value",
           "green_blue_pixel_value", "blue_pixel_value"]
SCALAR_FIELDS = {"exposure": "exposure", "analogue_gain": "gain",
                 "digital_gain": "dig", "vertical_blanking": "vblank",
                 "test_pattern": "pattern", "red_pixel_value": "pr",
                 "green_red_pixel_value": "pgr", "green_blue_pixel_value": "pgb",
                 "blue_pixel_value": "pb"}

L = threading.RLock()
S = {
    "jpg": None, "sq": 0, "cond": threading.Condition(L),
    "note": "starting...", "mode": DEFAULT_MODE, "phase": "?",
    "scale": "half", "quality": 80, "wb": 1, "gamma": 1,
    "fps": 0.0, "frames": 0, "t_last": 0.0, "line_us": None,
    "backend": "?",
    "want": dict(DEF_WANT), "ctl": {}, "ranges": {},
    "stream_w": 0, "stream_h": 0,
}
JOBS = []   # truthy entries mean: tear down and reconfigure (mode/flip change)


def note(t):
    with L:
        S["note"] = t
        print("camkit:", t, flush=True)


def push_job(desc):
    with L:
        JOBS.append(True)
        S["note"] = "reconfiguring: %s" % desc


def run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return r.returncode, (r.stdout or b"").decode(errors="replace")
    except Exception as e:
        return 99, repr(e)


def sensor_set(**kv):
    """Set sensor controls - each -c as its own arg (comma form fails here)."""
    args = []
    for k, v in kv.items():
        args += ["-c", "%s=%d" % (k, int(v))]
    return run(["/usr/bin/v4l2-ctl", "-d", SENS] + args)


def get_ctl():
    """Read control values + ranges from the kernel driver."""
    rc, out = run(["/usr/bin/v4l2-ctl", "-d", SENS, "-l"])
    vals, ranges = {}, {}
    for line in out.splitlines():
        m = re.match(r"\s*([a-zA-Z_0-9]+)\s+0x[0-9a-f]+\s+\(\S+\)\s*:", line)
        if not m:
            continue
        name = m.group(1)
        mm = re.search(r"min=(-?\d+)\s+max=(-?\d+)", line)
        vm = re.search(r"(?:^|\s)value=(-?\d+)", line)
        if mm:
            ranges[name] = (int(mm.group(1)), int(mm.group(2)))
        if vm:
            vals[name] = int(vm.group(1))
    return vals, ranges


def media_fmt(W, H):
    """Propagate a sensor format through the whole capture chain."""
    errors = []
    for ent in ("imx219 4-0010",
                "cdns_csi2rx.30101000.csi-bridge",
                "30102000.ticsi2rx"):
        q = '"%s":0 [fmt:SRGGB10_1X10/%dx%d field:none]' % (ent, W, H)
        rc, out = run(["/usr/bin/media-ctl", "-d", MEDIA, "-V", q])
        if rc:
            errors.append(out.strip()[:120])
    rc, out = run(["/usr/bin/v4l2-ctl", "-d", CAP,
                   "--set-fmt-video=width=%d,height=%d,pixelformat=RG10" % (W, H)])
    if rc:
        errors.append(out.strip()[:120])
    if errors:
        raise RuntimeError("pipeline config failed: " + " | ".join(errors))


def read1(fd, size, timeout=1.5):
    if not select.select([fd], [], [], timeout)[0]:
        return None
    try:
        data = os.read(fd, size)
    except BlockingIOError:
        return None
    return data if len(data) == size else None


def identify_phase(cap, W, H):
    """Re-verify the Bayer phase with the sensor's solid-red test pattern
    (proven recipe: only the correct OpenCV code yields R=255, G=B=0).
    Needed at every open because flips/modes change the phase."""
    sensor_set(test_pattern=2, red_pixel_value=1023,
               green_red_pixel_value=0, green_blue_pixel_value=0,
               blue_pixel_value=0)
    best = None
    for _ in range(10):
        r = fetch_timeout(cap, 1.5)
        if not r:
            continue
        arr, idx = r
        cap.put(idx)
        b8 = (arr >> 2).astype(np.uint8).reshape(H, W)
        for nm, code in DEBAYER.items():
            x = cv2.cvtColor(b8, code)
            B, G, R = [float(x[..., i].mean()) for i in range(3)]
            ratio = R / max(1.0, (B + G) / 2.0)
            if best is None or ratio > best[1]:
                best = (nm, ratio)
        cap.put(idx)
    # restore the user's pattern settings
    with L:
        w = dict(S["want"])
    sensor_set(test_pattern=w.get("test_pattern", 0),
               red_pixel_value=w.get("red_pixel_value", 1023),
               green_red_pixel_value=w.get("green_red_pixel_value", 1023),
               green_blue_pixel_value=w.get("green_blue_pixel_value", 1023),
               blue_pixel_value=w.get("blue_pixel_value", 1023))
    if best is None or best[1] < 3.0:
        raise RuntimeError("Bayer phase identification failed (ratio %s)"
                           % ("%.2f" % best[1] if best else "n/a"))
    return best[0]


def apply_want(ranges):
    """Clamp desired controls to the current driver ranges and push them."""
    with L:
        want = dict(S["want"])
    out = {}
    for k in SCALARS + ["horizontal_flip", "vertical_flip"]:
        lo, hi = ranges.get(k, RANGE_FALLBACK[k])
        v = int(min(max(want.get(k, DEF_WANT[k]), lo), hi))
        want[k] = v
        out[k] = v
    rc, o = sensor_set(**out)
    with L:
        S["want"] = want
    if rc:
        raise RuntimeError("sensor set rejected: %s" % o.strip()[-160:])
    return want


def stream_dims(W, H):
    f = SCALES.get(S["scale"], 0.5)
    return (max(2, (int(W * f) // 2) * 2), max(2, (int(H * f) // 2) * 2))


def reconfigure():
    """Full teardown/rebuild of the capture session (mode or flips changed)."""
    with L:
        mode = S["mode"]
        S["note"] = "configuring %s ..." % mode
    W, H = MODES[mode]
    media_fmt(W, H)
    _, ranges = get_ctl()
    merged = dict(RANGE_FALLBACK)
    merged.update(ranges)
    apply_want(merged)
    with L:
        S["ranges"] = merged
        S["ctl"], _ = get_ctl()
    fd = os.open(CAP, os.O_RDWR | os.O_NONBLOCK)   # RDWR: STREAMON needs it
    npix = W * H
    backend = None
    cap = None
    errs = []

    def _release_queue():
        """vb2 queue on this driver is shared; free leftover REQBUFS state
        so the read fallback (and later S_FMT) does not hit EBUSY."""
        try:
            rb = v4l2_requestbuffers()
            rb.count, rb.type, rb.memory = 0, _V4L2_BUF_TYPE_VIDEO_CAPTURE, _V4L2_MEMORY_MMAP
            libc.ioctl(fd, IO_REQBUFS, ctypes.byref(rb))
        except Exception:
            pass

    try:
        for bcls in (PipeCap, MMapCap):
            try:
                cap = bcls(fd, npix, W, H)
                backend = bcls.name
                break
            except Exception as e:
                errs.append("%s:%s" % (bcls.name, str(e)[:70]))
        if cap is None:
            _release_queue()
            backend = "read"
            cap = ReadCap(fd, npix)
            note("fast backends unavailable - read fallback (%s)"
                 % " | ".join(errs))
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        raise
    phase = identify_phase(cap, W, H)
    vals, _ = get_ctl()
    # line time straight from the kernel-reported pixel_rate + horizontal
    # blanking - deterministic within one driver (no warmup dt statistics)
    pw = vals.get("pixel_rate", PIXEL_RATE)
    hb = vals.get("horizontal_blanking")
    line_us = ((W + hb) * 1e6 / pw) if hb is not None else None
    with L:
        S["phase"] = phase
        S["backend"] = backend
        S["line_us"] = line_us
        S["ctl"] = vals
        S["stream_w"], S["stream_h"] = stream_dims(W, H)
    return fd, cap, W, H


def capture_loop():
    note("camkit starting: opening CSI pipeline...")
    fd = None
    cap = None
    while True:
        try:
            with L:
                JOBS.clear()
            # teardown of the previous session FIRST: this driver ignores
            # writes (flips!) while a node streams, and reconfigure needs
            # the device fully idle anyway
            if cap is not None:
                try:
                    cap.stop()
                except Exception:
                    pass
                cap = None
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
                fd = None
            fd, cap, W, H = reconfigure()
            size = W * H * 2
            with L:
                sw, sh = S["stream_w"], S["stream_h"]
                backend = S["backend"]
                S["note"] = "%s live | phase %s verified | view %dx%d (%s)" \
                            % (S["mode"], S["phase"], sw, sh, backend)
            last_ok = time.time()
            fps = 0.0
            fidx = 0
            while True:
                with L:
                    if JOBS:
                        JOBS.clear()
                        break
                    sw, sh = S["stream_w"], S["stream_h"]
                    q = int(S["quality"])
                    phase = S["phase"]
                r = fetch_timeout(cap, 2.0)
                if r is None:
                    if time.time() - last_ok > 8.0:
                        raise RuntimeError("stream stalled (no frames 8 s)")
                    continue
                arr, idx = r
                now = time.time()
                dt = now - last_ok
                last_ok = now
                if dt > 0 and fidx >= 3:   # fidx<3: ring-refill bursts lie
                    fps = (1.0 / dt) if fps <= 0.0 else fps * 0.9 + (0.1 / dt)
                b8 = (arr >> 2).astype(np.uint8).reshape(H, W)
                # b8 now holds copies of every pixel we need; the DMA
                # buffer is safe to recycle for the next sensor frame
                cap.put(idx)
                bgr = cv2.cvtColor(b8, DEBAYER.get(phase, DEBAYER["BG"]))
                if (sw, sh) != (W, H):
                    bgr = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
                with L:
                    wb_on = int(S["wb"]) != 0
                    gamma_on = int(S["gamma"]) != 0
                if wb_on:
                    # gray-world AWB on a sparse sample; green is the anchor
                    m = bgr[::16, ::16].astype("f4").mean(axis=(0, 1))
                    if m[1] > 8.0 and m[0] > 4.0 and m[2] > 4.0:
                        t = np.clip(m[1] / np.maximum(m, 0.5), 0.6, 2.2)
                        WB_G[0] = 0.95 * WB_G[0] + 0.05 * float(t[0])
                        WB_G[1] = 0.95 * WB_G[1] + 0.05 * float(t[1])
                        WB_G[2] = 0.95 * WB_G[2] + 0.05 * float(t[2])
                if wb_on or gamma_on:
                    bgr = cv2.LUT(bgr, build_lut(wb_on, gamma_on))
                ok, jpg = cv2.imencode(".jpg", bgr,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), q])
                if not ok:
                    raise RuntimeError("jpeg encode failed")
                with L:
                    S["jpg"] = jpg.tobytes()
                    S["sq"] += 1
                    S["frames"] += 1
                    S["fps"] = fps
                    S["t_last"] = now
                    S["cond"].notify_all()
                fidx += 1
        except Exception as e:
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
                fd = None
            note("camera problem: %s - retrying every 2 s" % str(e)[:130])
            time.sleep(2.0)


def clamp_int(name, raw, merged):
    lo, hi = merged.get(name, RANGE_FALLBACK[name])
    try:
        v = int(float(raw))
    except Exception:
        return None
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ------------------------------- HTTP ---------------------------------------

def status_json():
    with L:
        d = {"fps": round(S["fps"], 2), "frames": S["frames"],
             "frame_age_ms": int((time.time() - S["t_last"]) * 1000)
                             if S["t_last"] else -1,
             "mode": S["mode"], "phase": S["phase"], "backend": S["backend"],
             "scale": {"half": "half", "34": "3/4", "full": "full"}[S["scale"]],
             "stream": [S["stream_w"], S["stream_h"]],
             "quality": S["quality"], "line_us": S["line_us"],
             "wb": S["wb"], "gamma": S["gamma"],
             "note": S["note"], "want": dict(S["want"]),
             "ctl": dict(S["ctl"]), "ranges": {k: list(v) for k, v in S["ranges"].items()}}
    return d


def page():
    with L:
        w = dict(S["want"])
        mode = S["mode"]
        scale = S["scale"]
        quality = S["quality"]
        phase = S["phase"]
        fps = S["fps"]
        frames = S["frames"]
        line_us = S["line_us"]
        note_t = S["note"]
        sw, sh = S["stream_w"], S["stream_h"]
    merged = dict(RANGE_FALLBACK)
    with L:
        merged.update(S["ranges"])

    def rg(name):
        return merged.get(name, RANGE_FALLBACK[name])

    def num(name, field, label, step=1):
        lo, hi = rg(name)
        v = w.get(name, DEF_WANT[name])
        return ('<div class="row"><label>%s</label>'
                '<input type="number" name="%s" min="%d" max="%d" step="%d" value="%d">'
                '<span class="clr">driver range %d..%d</span></div>'
                % (label, field, lo, hi, step, v, lo, hi))

    mode_opts = []
    for m in MODES:
        sel = " selected" if m == mode else ""
        mode_opts.append('<option value="%s"%s>%s  (%s)</option>'
                         % (m, sel, m, MODE_NOTES[m]))
    pat_opts = []
    for v, t in ((0, "0 - Disabled (real scene)"), (1, "1 - Color bars"),
                 (2, "2 - Solid color"), (3, "3 - Grey color bars"),
                 (4, "4 - PN9")):
        sel = " selected" if v == w.get("test_pattern", 0) else ""
        pat_opts.append('<option value="%d"%s>%s</option>' % (v, sel, t))
    sc_opts = []
    for sv, t in (("half", "half (fast)"), ("34", "3/4"), ("full", "full")):
        sel = " selected" if sv == scale else ""
        sc_opts.append('<option value="%s"%s>%s</option>' % (sv, sel, t))
    hf = " checked" if w.get("horizontal_flip", 0) else ""
    vf = " checked" if w.get("vertical_flip", 0) else ""
    est = ""
    if line_us:
        vb = w.get("vertical_blanking", 683)
        H = MODES[mode][1]
        est = "~%.1f fps from vblank" % (1e6 / ((H + vb) * line_us))

    head = ('<div class="wrap">'
            '<div class="panel">'
            '<img id="v" src="/mjpg" alt="live view">'
            '<div class="stat"><a href="/frame.jpg" target="_blank">'
            'open snapshot</a> &nbsp;|&nbsp; view <b>%dx%d</b> '
            '&bull; mode <b>%s</b> &bull; bayer phase <b>%s</b> '
            '(verified at open)</div></div>'
            % (sw, sh, mode, phase))
    form = ['<div class="panel"><form method="POST" action="/set">',
            '<h2>Switches &amp; frame rate</h2>',
            num("exposure", "exposure", "exposure (lines)"),
            num("analogue_gain", "gain", "analogue gain"),
            num("digital_gain", "dig", "digital gain"),
            num("vertical_blanking", "vblank", "vertical blanking")]
    form.append(
        '<div class="row"><label>frame-rate preset</label>'
        '<select name="fpreset">'
        '<option value="0">keep (use vblank above)</option>'
        '<option value="30">30 fps</option><option value="25">25 fps</option>'
        '<option value="20">20 fps</option><option value="15">15 fps</option>'
        '<option value="10">10 fps</option><option value="5">5 fps</option>'
        '<option value="2">2 fps</option>'
        '</select><span class="clr">%s</span></div>' % est)
    form.append('<h2>Test pattern (replaces scene)</h2>')
    form.append('<div class="row"><label>test pattern</label>'
                '<select name="pattern">%s</select></div>'
                % "".join(pat_opts))
    form.append(num("red_pixel_value", "pr", "solid red pixel"))
    form.append(num("green_red_pixel_value", "pgr", "green A pixel"))
    form.append(num("green_blue_pixel_value", "pgb", "green B pixel"))
    form.append(num("blue_pixel_value", "pb", "solid blue pixel"))
    form.append('<div class="clr">pixel values only matter with patterns 2/3.'
                '</div>')
    form.append('<h2>Orientation</h2>')
    form.append('<div class="row"><label>flip</label>'
                '<label class="cbx"><input type="checkbox" name="hflip" '
                'value="1"%s> horizontal</label>'
                '<label class="cbx"><input type="checkbox" name="vflip" '
                'value="1"%s> vertical</label></div>' % (hf, vf))
    form.append('<h2>View</h2>')
    form.append('<div class="row"><label>sensor mode</label>'
                '<select name="mode">%s</select></div>'
                % "".join(mode_opts))
    form.append('<div class="row"><label>view size</label>'
                '<select name="scale">%s</select></div>'
                % "".join(sc_opts))
    form.append('<div class="row"><label>view JPEG quality</label>'
                '<input type="number" name="quality" min="30" max="95" '
                'value="%d"><span class="clr">30..95</span></div>' % quality)
    form.append('<h2>Color</h2>')
    form.append('<div class="row"><label>color aids</label>'
                '<label class="cbx"><input type="checkbox" name="wb" '
                'value="1"%s> white balance</label>'
                '<label class="cbx"><input type="checkbox" name="gamma" '
                'value="1"%s> shadow lift</label>'
                '<span class="clr">WB fixes the raw green cast '
                '(gray-world, auto)</span></div>'
                % (" checked" if int(S.get("wb", 1)) else "",
                   " checked" if int(S.get("gamma", 0)) else ""))
    form.append('<div class="row"><label></label>'
                '<button type="submit">Apply</button>'
                '<span class="clr">flips &amp; mode switch take ~1-2 s;</span>'
                '</div>')
    form.append('</form>'
                '<div class="note" id="s_note">note: %s</div>' % note_t)
    form.append('</div>')
    html = PAGE_HEAD + head + "".join(form) + PAGE_FOOT
    return html


PAGE_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<title>Camera Lab - SK-AM62 (IMX219)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{background:#0e1116;color:#cfd8dc;font-family:system-ui,Segoe UI,sans-serif;padding:12px;margin:0;font-size:14px}
 h1{font-size:18px;margin:0 0 10px;color:#fff}
 h2{font-size:13px;margin:14px 0 4px;color:#8fb8c9;border-bottom:1px solid #2a333c;padding-bottom:3px}
 .wrap{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;max-width:1150px}
 .panel{background:#161b21;border:1px solid #2a333c;border-radius:9px;padding:10px 12px}
 #v{max-width:100%;display:block;border:1px solid #333;background:#000;border-radius:4px}
 .stat{font-size:12px;color:#7fc97f;margin-top:6px;font-family:monospace}
 .stat a{color:#6fb3ff;text-decoration:none}
 .row{display:flex;gap:8px;align-items:center;margin:6px 0}
 .row label{min-width:150px;color:#8fb8c9;font-size:13px}
 .clr{font-size:11px;color:#5c6a72;margin-left:6px}
 .cbx{min-width:0;margin-right:10px;color:#cfd8dc}
 input[type=number],select{background:#0b0f14;color:#e6edf3;border:1px solid #333;padding:3px 6px;min-width:110px}
 input[type=number]{width:84px}
 button{background:#1f6feb;color:#fff;border:0;padding:7px 16px;border-radius:5px;cursor:pointer;font-size:14px}
 .note{font-size:12px;color:#e3b341;margin-top:8px;max-width:560px}
 .foot{font-size:11px;color:#5c6a72;margin-top:12px;max-width:900px;line-height:1.5}
</style></head><body>
<h1>Camera Lab - SK-AM62 CSI (Arducam IMX219) <span id="dot">&#9679;</span></h1>
"""

PAGE_FOOT = """
<div class="foot">
Notes: every Apply writes the values straight to the sensor (live). The bayer
phase is re-verified at every (re)open with the sensor's solid-red pattern, so
flips and mode switches cannot silently scramble colors. FPS presets convert to
vertical blanking via a self-measured line time; the live fps meter is the
ground truth. Stopping this service hands the camera back to bird detection
(systemctl stop camkit); starting it stops birdboard automatically.
</div>
<script>
function tset(id,t){document.getElementById(id).textContent=t}
var lastOK=0;
function tick(){ if(document.hidden){return;}
 fetch('/status').then(function(r){return r.json()}).then(function(j){
  lastOK=Date.now();
  tset('s_note','note: '+j.note+'  |  '+j.fps.toFixed(1)+' fps  |  frames '+j.frames+'  |  phase '+j.phase+'  |  '+j.stream[0]+'x'+j.stream[1]);
  document.getElementById('dot').style.color = (j.frame_age_ms < 3000 && j.frame_age_ms >= 0) ? '#7fc97f' : '#e3b341';
 }).catch(function(){ document.getElementById('dot').style.color='#e3b341'; });
}
setInterval(tick,2000); tick();
</script></body></html>
"""


class CK(BaseHTTPRequestHandler):
    server_version = "camkit/1.0"

    def log_message(self, fmt, *args):
        pass

    def _hdr(self, code, ctype, extra=b""):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(extra)))
        self.end_headers()
        return extra

    def _redirect(self):
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if p == "/":
                self._send_bytes(page().encode(), "text/html; charset=utf-8")
            elif p == "/status":
                self._send_bytes(json.dumps(status_json()).encode(),
                                 "application/json")
            elif p == "/frame.jpg":
                with L:
                    j = S["jpg"]
                if not j:
                    self._hdr(503, "text/plain", b"no frame yet")
                else:
                    self._send_bytes(j, "image/jpeg")
            elif p == "/mjpg":
                self._mjpg()
            elif p == "/readctl":
                vals, rng = get_ctl()
                with L:
                    S["ctl"] = vals
                    S["note"] = "control values re-read from kernel"
                self._redirect()
            elif p == "/ping":
                self._hdr(200, "text/plain", b"camkit-ok")
            else:
                self._hdr(404, "text/plain", b"not found")
        except BrokenPipeError:
            pass

    def _send_bytes(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _mjpg(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=camkitframe")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        miss = 0
        try:
            while True:
                jpg = None
                with L:
                    if S["sq"] != last:
                        jpg = S["jpg"]
                        last = S["sq"]
                        miss = 0
                    else:
                        S["cond"].wait(2.5)
                if jpg is None:
                    miss += 1
                    if miss > 8:
                        return
                    continue
                self.wfile.write(b"--camkitframe\r\nContent-Type: image/jpeg"
                                 b"\r\nContent-Length: " +
                                 str(len(jpg)).encode() + b"\r\n\r\n" + jpg +
                                 b"\r\n")
        except Exception:
            return

    def do_POST(self):
        if self.path.split("?")[0] != "/set":
            self._hdr(404, "text/plain", b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", "0") or 0)
            f = parse_qs(self.rfile.read(n).decode(errors="replace"))
        except Exception:
            f = {}
        try:
            self._apply_set(f)
        except Exception as e:
            with L:
                S["note"] = "apply error: %s" % str(e)[:160]
        self._redirect()

    def _apply_set(self, f):
        with L:
            merged = dict(RANGE_FALLBACK)
            merged.update(S["ranges"])
            want = dict(S["want"])
            old_mode = S["mode"]
            old_h = want.get("horizontal_flip", 0)
            old_v = want.get("vertical_flip", 0)
            notes = []

        def iv(field):
            v = f.get(field)
            if not v:
                return None
            try:
                return int(float(v[0]))
            except Exception:
                return None

        # scalar sensor controls (one combined sensor_set call)
        kv = {}
        for cname, field in SCALAR_FIELDS.items():
            v = iv(field)
            if v is None:
                continue
            c = clamp_int(cname, v, merged)
            if c is None:
                continue
            want[cname] = c
            kv[cname] = c
        # fps preset -> vertical_blanking via self-measured line time
        fp = iv("fpreset")
        if fp and fp > 0:
            with L:
                line_us = S["line_us"]
                H = MODES[S["mode"]][1]
            if line_us:
                vb = int((1e6 / (fp * line_us)) - H)
                lo, hi = merged.get("vertical_blanking",
                                    RANGE_FALLBACK["vertical_blanking"])
                vb = max(lo, min(hi, vb))
                want["vertical_blanking"] = vb
                kv["vertical_blanking"] = vb
                notes.append("preset %d fps -> vblank %d" % (fp, vb))
            else:
                notes.append("fps preset needs calibration first "
                             "(let the stream run 2 s, retry)")
        if kv:
            rc, out = sensor_set(**kv)
            if rc:
                notes.append("driver rejected: %s" % out.strip()[-120:])
            else:
                vals, _ = get_ctl()
                with L:
                    S["ctl"] = vals
        # pure view settings (no sensor involvement)
        sc = (f.get("scale") or [None])[0]
        if sc in SCALES and sc != S["scale"]:
            with L:
                S["scale"] = sc
                W, H = MODES[S["mode"]]
                S["stream_w"], S["stream_h"] = stream_dims(W, H)
            notes.append("view %s" % {"half": "half", "34": "3/4",
                                      "full": "full"}[sc])
        qv = iv("quality")
        if qv is not None:
            with L:
                S["quality"] = max(30, min(95, qv))
        wbv = (f.get("wb") or [None])[0]
        if wbv is not None:
            on = 1 if wbv.strip().lower() in ("1", "on", "true") else 0
            with L:
                S["wb"] = on
            notes.append("white balance %s" % ("on" if on else "off"))
        gv = (f.get("gamma") or [None])[0]
        if gv is not None:
            on = 1 if gv.strip().lower() in ("1", "on", "true") else 0
            with L:
                S["gamma"] = on
            notes.append("shadow gamma %s" % ("on" if on else "off"))
        need_reconf = False
        mv = (f.get("mode") or [None])[0]
        with L:
            if mv in MODES and mv != S["mode"]:
                S["mode"] = mv
                need_reconf = True
        hraw = (f.get("hflip") or [""])[0].lower()
        vraw = (f.get("vflip") or [""])[0].lower()
        hv = 1 if hraw in ("1", "on", "true") else 0
        vv = 1 if vraw in ("1", "on", "true") else 0
        if hv != old_h or vv != old_v:
            want["horizontal_flip"] = hv
            want["vertical_flip"] = vv
            need_reconf = True
        with L:
            S["want"] = want
        if need_reconf:
            push_job("mode/flips change requested")
        else:
            with L:
                if notes:
                    S["note"] = "applied: " + "; ".join(notes)[:120]
                else:
                    S["note"] = "no changes in form"


def main():
    # pin all cores to performance: the A53 governor otherwise parks the
    # cores low and halves the whole pipeline unpredictably
    try:
        for c in sorted(__import__("glob").glob(
                "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")):
            with open(c, "w") as fh:
                fh.write("performance")
    except Exception:
        pass
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), CK)
    srv.daemon_threads = True
    print("camkit: http listening on :%d" % PORT, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
