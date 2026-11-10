"""
camera_bridge.py v3 - phone camera -> TI SK-AM62 bird-detection live viewer (PC side)

Usage (from venv):
    venv\\Scripts\\python.exe camera_bridge.py "http://PHONE_IP:4747/video"
    venv\\Scripts\\python.exe camera_bridge.py            # no arg -> laptop webcam

Serves on port 9000:
    /view      LIVE INFERENCE DASHBOARD (auto-updating + phone/board status)
    /stream    raw camera MJPEG stream
    /frame.jpg latest raw frame (the board wget's this each cycle)
    /det.jpg   latest annotated inference image (proxied from board)
    /meta      latest inference JSON from the board
    /phone     camera link status JSON (auto-reconnect info)
    /ping      health check
The phone link auto-reconnects every 2 seconds whenever it drops.
Also mirrors the board's result into Desktop\\Bird Detection Results\\live_det.jpg.
"""
import sys, time, threading, urllib.request, pathlib, json
import cv2
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

SOURCE = None
RETRY_EVERY = 2.0            # seconds between phone reconnect attempts

lock = threading.Lock()
latest = {"jpeg": None, "t": 0.0}            # raw frames
det = {"jpeg": None, "t": 0.0}               # annotated result from board
meta = {"json": b'{"n":null,"infer_ms":null}', "t": 0.0}
phone = {"connected": False, "last_ok": 0.0, "last_try": 0.0,
         "tries": 0, "src": None, "note": "starting"}

BOARD_LIVE = "http://192.168.1.22:8888/live_det.jpg"
BOARD_META = "http://192.168.1.22:8888/meta.json"

def desktop_dir():
    d = pathlib.Path.home() / "Desktop"
    if not d.exists():
        d = pathlib.Path.home() / "OneDrive" / "Desktop"
    out = d / "Bird Detection Results"
    out.mkdir(exist_ok=True)
    return out

def push_frames():
    """Background: read camera continuously; auto-reconnect every 2 s when down."""
    while True:
        try:
            if SOURCE and SOURCE.startswith("http"):
                with lock:
                    phone.update({"connected": False,
                                  "last_try": time.time(),
                                  "note": "connecting to phone..."})
                cap = cv2.VideoCapture(SOURCE)
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("stream ended/blocked")
                    h, w = frame.shape[:2]
                    if w > 1280:
                        frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
                    okj, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if okj:
                        with lock:
                            latest["jpeg"], latest["t"] = enc.tobytes(), time.time()
                            phone.update({"connected": True, "last_ok": time.time(),
                                          "tries": 0, "note": "phone: LIVE"})
                    time.sleep(0.03)
            else:
                with lock:
                    phone.update({"src_note": "laptop webcam",
                                  "last_try": time.time(),
                                  "note": "opening webcam..."})
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("webcam blocked")
                    okj, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if okj:
                        with lock:
                            latest["jpeg"], latest["t"] = enc.tobytes(), time.time()
                            phone.update({"connected": True, "last_ok": time.time(),
                                          "tries": 0, "note": "webcam: LIVE"})
                    time.sleep(0.05)
        except Exception as e:
            wait = RETRY_EVERY
            with lock:
                phone["tries"] += 1
                phone["last_try"] = time.time()
                phone["connected"] = False
                if SOURCE and SOURCE.startswith("http"):
                    phone["note"] = "phone NOT connected - retrying every 2 s"
                else:
                    phone["note"] = "webcam not available - retrying every 2 s"
            print("camera down (%s) - retry in %.0f s" % (e, wait))
        if SOURCE and SOURCE.startswith("http"):
            try:
                cap.release()
            except Exception:
                pass
        time.sleep(RETRY_EVERY)

def pull_results():
    """Background: poll board for annotated result + metadata."""
    dest = desktop_dir() / "live_det.jpg"
    last_written = -1
    while True:
        try:
            data = urllib.request.urlopen(BOARD_LIVE, timeout=6).read()
            if len(data) > 1000:
                with lock:
                    det["jpeg"], det["t"] = data, time.time()
                if len(data) != last_written:
                    tmp = dest.with_suffix(".new")
                    tmp.write_bytes(data)
                    tmp.replace(dest)
                    last_written = len(data)
        except Exception:
            pass
        try:
            m = urllib.request.urlopen(BOARD_META, timeout=4).read()
            json.loads(m.decode("utf-8", "ignore"))
            with lock:
                meta["json"], meta["t"] = m, time.time()
        except Exception:
            pass
        time.sleep(1.2)

VIEW_HTML = b"""<!doctype html><html><head><title>Live Bird Detection - TI AM62 Board</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{background:#0e1116;color:#dbe2ef;font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#8fa1c7;font-size:13px;margin-bottom:12px}
.grid{display:flex;flex-wrap:wrap;gap:12px}
.card{background:#161b24;border:1px solid #2a3245;border-radius:10px;padding:8px}
.card h3{margin:2px 0 6px;font-size:14px;color:#9ab4e8}
img{max-width:96vw;border-radius:8px;background:#000}
#det{max-height:70vh}
.stats{margin-top:12px;font-size:13px;color:#8fa1c7;line-height:1.9}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#888;margin-right:5px}
.dot.ok{background:#2ecc71}
.dot.warn{background:#e67e22}
.dot.bad{background:#e74c3c}
.warnbox{background:#3a2a18;border:1px solid #7a5a2a;color:#f5c06a;border-radius:8px;padding:6px 10px;margin-top:10px;font-size:13px;display:none}
#ph{color:#8fa1c7;font-size:13px;padding:6px}
a{color:#66aaff}
</style></head><body>
<h1>Live bird detection - TI SK-AM62 board</h1>
<div class="sub">phone camera &rarr; PC bridge &rarr; board YOLOv8 inference &rarr; this page &middot; any device on the Wi-Fi: http://192.168.1.5:9000/view</div>
<div class="grid">
 <div class="card"><h3>Camera (live stream)</h3><img id="cam" src="/stream" width="620" alt="raw camera stream"></div>
 <div class="card"><h3>Board inference result</h3><div id="ph">waiting for first board result...</div><img id="det" src="" width="620" alt="inference result"></div>
</div>
<div class="stats">
 <span id="pdot" class="dot"></span><span id="phone">phone: checking...</span> &nbsp;|&nbsp;
 <span id="dot" class="dot"></span><span id="board">board: connecting...</span> &nbsp;|&nbsp;
 birds: <b id="n">-</b> &nbsp;|&nbsp; inference: <b id="ms">-</b> ms &nbsp;|&nbsp; result update: <b id="t">-</b> &nbsp;|&nbsp; <a href="/stream" target="_blank">full-screen raw feed</a>
</div>
<div class="warnbox" id="warn">Camera is not connected - the bridge is retrying every 2 seconds. Open the DroidCam app on your phone and tap "WiFi Server", and it will come back automatically.</div>
<script>
var det=document.getElementById('det'),ph=document.getElementById('ph'),dot=document.getElementById('dot'),bt=document.getElementById('board');
var pdot=document.getElementById('pdot'),pt=document.getElementById('phone'),warn=document.getElementById('warn');
var cam=document.getElementById('cam');
cam.onerror=function(){setTimeout(function(){cam.src='/stream?'+Date.now();},1500);};
det.onerror=function(){det.removeAttribute('src');};
det.onload=function(){ph.style.display='none';};
setInterval(function(){
  var probe=new Image();
  probe.onload=function(){det.src='/det.jpg?'+Date.now();};
  probe.onerror=function(){};
  probe.src='/det.jpg?'+Date.now();
},1400);
var lastOk=0,lastPh=0;
setInterval(function(){
  fetch('/meta',{cache:'no-store'}).then(function(r){return r.json();}).then(function(j){
    lastOk=Date.now();
    dot.className='dot ok'; bt.textContent='board: LIVE';
    document.getElementById('n').textContent=(j.n==null?'-':j.n);
    document.getElementById('ms').textContent=(j.infer_ms==null?'-':Math.round(j.infer_ms));
    document.getElementById('t').textContent=new Date().toLocaleTimeString();
  }).catch(function(){});
},1500);
setInterval(function(){if(Date.now()-lastOk>7000){dot.className='dot';bt.textContent='board: reconnecting...';}},2000);
setInterval(function(){
  fetch('/phone',{cache:'no-store'}).then(function(r){return r.json();}).then(function(j){
    lastPh=Date.now();
    var src=(j.source&&j.source.indexOf('http')!==0)?j.source:'phone';
    if(j.connected){
      pdot.className='dot ok';
      pt.textContent=src+': LIVE (last frame '+(j.frame_age_ms/1000).toFixed(1)+' s ago)';
      warn.style.display='none';
    } else {
      pdot.className=(j.retries>0?'dot warn':'dot');
      pt.textContent=src+': not connected - retrying every '+j.retry_s+' s (attempt '+j.retries+')';
      warn.style.display='block';
    }
  }).catch(function(){});
},1500);
setInterval(function(){if(Date.now()-lastPh>7000){pdot.className='dot bad';pt.textContent='phone bridge: offline';}},2000);
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/frame.jpg":
            with lock: jpg = latest["jpeg"]
            if jpg: self.send(jpg, "image/jpeg")
            else: self.send_error(404)
        elif self.path == "/det.jpg":
            with lock: jpg = det["jpeg"]
            if jpg: self.send(jpg, "image/jpeg")
            else: self.send_error(404)
        elif self.path == "/meta":
            with lock: j = meta["json"]
            self.send(j, "application/json")
        elif self.path == "/phone":
            with lock:
                p = dict(phone)
                p["source"] = SOURCE or "laptop webcam"
                body = json.dumps({
                    "connected": p["connected"],
                    "frame_age_ms": int((time.time() - p["last_ok"]) * 1000) if p["last_ok"] else None,
                    "retries": p["tries"],
                    "retry_s": int(RETRY_EVERY),
                    "last_try_age_ms": int((time.time() - p["last_try"]) * 1000) if p["last_try"] else None,
                    "note": p["note"],
                    "source": p["source"],
                }).encode()
            self.send(body, "application/json")
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=jpegframe")
            self.end_headers()
            try:
                while True:
                    with lock: jpg = latest["jpeg"]
                    if jpg:
                        self.wfile.write(b"--jpegframe\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.4)
            except Exception:
                pass
        elif self.path in ("/view", "/"):
            self.send(VIEW_HTML, "text/html; charset=utf-8")
        elif self.path == "/ping":
            self.send(b"bridge-ok", "text/plain")
        else:
            self.send_error(404)

if __name__ == "__main__":
    SOURCE = sys.argv[1] if len(sys.argv) > 1 else None
    with lock:
        phone["src"] = SOURCE
        phone["source"] = SOURCE or "laptop webcam"
    threading.Thread(target=push_frames, daemon=True).start()
    threading.Thread(target=pull_results, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 9000), Handler)
    print("Bridge v3 running on http://0.0.0.0:9000  (source = %s)" % (SOURCE or "laptop webcam"))
    print("Live inference dashboard:  http://192.168.1.5:9000/view")
    srv.serve_forever()
