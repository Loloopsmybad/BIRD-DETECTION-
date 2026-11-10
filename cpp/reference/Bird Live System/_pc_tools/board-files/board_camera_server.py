#!/usr/bin/env python3
# board_camera_server.py - ALL-ON-BOARD bird pipeline for TI SK-AM62
#
#   camera source  (phone DroidCam MJPEG now, CSI/any MJPEG later)
#        |  raw JPEG bytes read directly over WiFi by this process
#        v  fresh frames -> board_detection.run() IN-PROCESS (cached ONNX net)
#        |  boxes drawn on the board, live_det.jpg + meta.json written on board
#        v  this server on :8888 hosts: dashboard, raw frames, det image,
#           metadata JSON, camera status (auto-reconnect every 2 s shown live)
#
# Efficiency notes: in-process inference (no subprocess per frame), ONNX net
# cached at first load, frames processed once, light dashboard that stops
# polling while hidden / camera-down, frame-stall watchdog -> reconnect.
#
# Camera source is read from /home/root/bird/camera_url.txt each reconnect
# attempt, so you can change it without restarting this service.
# Later for the CSI camera: write "/dev/video0" there (supported source type).

import os, sys, time, json, threading, http.client
from urllib.parse import urlsplit
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

BIRD   = "/home/root/bird"
URLFILE= os.path.join(BIRD, "camera_url.txt")
FRAME  = os.path.join(BIRD, "frame.jpg")
DET    = os.path.join(BIRD, "live_det.jpg")
META   = os.path.join(BIRD, "meta.json")

RETRY_EVERY     = 2.0     # seconds between camera reconnect attempts
CONNECT_TIMEOUT = 8       # seconds for the first HTTP connect
STREAM_TIMEOUT  = 20      # seconds with no bytes -> reconnect
FRAME_STALL     = 15      # seconds with NO complete jpeg frame -> reconnect
                          # (the phone can keep the socket alive but stop
                          #  streaming, e.g. DroidCam when the screen locks)

lock = threading.Lock()
S = {
    "jpeg": None, "jpeg_t": 0.0, "frames": 0,
    "meta_txt": b'{"n":null,"infer_ms":null,"err":"warming up"}',
    "last_infer_t": 0.0,
    "cam": dict(url="", connected=False, attempts=0, note="starting",
                last_try=0.0, last_ok=0.0),
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
    """
    global BD
    done_t = 0.0
    while True:
        time.sleep(0.25)
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
            d = BD.run("best.onnx", "frame.jpg", 640, 0.25, 0.45, True,
                       verbose=False, jpeg_bytes=jpg)
        except Exception as e:
            d = {"n": None, "infer_ms": None,
                 "err": "detection error: %s" % str(e)[:80]}
        body = json.dumps(d).encode()
        mt = META + ".tmp"
        try:
            f = open(mt, "wb"); f.write(body); f.close()
            os.replace(mt, META)
        except Exception:
            pass
        with lock:
            S["meta_txt"] = body
            S["last_infer_t"] = time.time()
        fp = os.path.join(BIRD, "frame_det.jpg")
        try:
            if os.path.exists(fp):
                os.replace(fp, DET)
        except Exception:
            pass

HTML = b"""<!doctype html><html><head><title>Live Bird Detection (all on TI board)</title>
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
</style></head><body>
<h1>Live bird detection - running on the TI SK-AM62 board</h1>
<div class="sub">camera &rarr; board capture &rarr; board YOLOv8 &rarr; boxes drawn on board &middot; page served by the board &middot; <a href="/stream">full-screen stream</a></div>
<div class="grid">
 <div class="card"><h3>Camera</h3><img id="cam" alt="camera"></div>
 <div class="card"><h3>Detection (boxes drawn on board)</h3><div id="ph">waiting for first result...</div><img id="det" alt="result"></div>
</div>
<div class="stats">
 <span id="pdot" class="dot"></span><span id="phone">camera: checking...</span> &nbsp;|&nbsp;
 birds: <b id="n">-</b> &nbsp;|&nbsp; infer: <b id="ms">-</b> ms &nbsp;|&nbsp; updated: <b id="t">-</b>
</div>
<div class="warnbox" id="warn"></div>
<script>
var cam=document.getElementById('cam'),det=document.getElementById('det'),ph=document.getElementById('ph');
var pdot=document.getElementById('pdot'),pt=document.getElementById('phone'),warn=document.getElementById('warn');
var camLive=false;   // drives image refresh ONLY while frames actually flow
cam.onerror=function(){};
det.onerror=function(){};
det.onload=function(){ph.style.display='none';};
/* camera panel: one image fetch per tick (no constant MJPEG upload),
   and it stops entirely whenever the camera link is down */
setInterval(function(){ if(camLive&&!document.hidden){cam.src='/frame.jpg?'+Date.now();} },1000);
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
  }).catch(function(){});
},2000);
</script></body></html>"""

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
    def do_GET(self):
        if self.path in ("/", "/view", "/index.html"):
            return self.emit(200, HTML, "text/html; charset=utf-8")
        if self.path.startswith("/frame.jpg"):
            with lock:
                jpg = S["jpeg"]
            if not jpg:
                return self.emit(404, b'{"err":"no frame yet"}')
            return self.emit(200, jpg, "image/jpeg")
        if self.path.startswith("/det.jpg") or self.path.startswith("/live_det.jpg"):
            try:
                jpg = open(DET, "rb").read()
                return self.emit(200, jpg, "image/jpeg")
            except Exception:
                return self.emit(404, b'{"err":"no result yet"}')
        if self.path in ("/meta.json", "/meta"):
            with lock:
                t = S["meta_txt"]
            return self.emit(200, t)
        if self.path == "/phone":
            with lock:
                c = dict(S["cam"])
                frames = S["frames"]
                jt = S["jpeg_t"]
            fa = int((time.time() - jt) * 1000) if jt else None
            body = json.dumps({
                "connected": c["connected"], "note": c["note"], "url": c["url"],
                "attempts": c["attempts"], "retry_s": int(RETRY_EVERY),
                "frame_age_ms": fa, "frames": frames,
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
                    time.sleep(1.0)   # throttled: keeps full-screen viewers cheap
            except Exception:
                pass
        if self.path == "/ping":
            return self.emit(200, b"board-ok", "text/plain")
        self.emit(404, b"")

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=infer_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    print("Board server live on http://0.0.0.0:8888/  (camera: see %s)" % URLFILE)
    srv.serve_forever()
