# Camera & Detection Pipeline — Technical Documentation

Technical reference for the **CSI camera pipeline** shared by every
component of this repo, plus the keep-the-honest-numbers record of how
each performance figure was measured. Everything runs on the
**TI SK-AM62** board (4× Cortex-A53, kernel 5.10 vendor tree) with the
**Arducam V2.2 (IMX219)** CSI camera on `/dev/video0` (RG10, 2 bytes/pixel
raw Bayer).

**Repo layout map**

| component | repo folder | board home | port | purpose |
|---|---|---|---|---|
| **C birdboard — production** | [cpp\](cpp/) | `/home/root/birdcpp` | **8888** | bird detection, live-media-only, native ORT (`birdcpp.service`) |
| python birdboard — parity/handback | [python\](python/) | `/home/root/bird` | 8888 | original all-python system (`birdboard.service`) |
| camera tuning lab | [camkit\](camkit/) | `/home/root/camkit` | 8889 | interactive sensor/phase/WB lab (`camkit.service`) |
| shared frame copier core | [frame_pumper\](frame_pumper/) | `/home/root/camkit`, `/home/root/bird` (`frame_pumper`) | - | fork/exec stage copying 4 MB DMA frames out of uncached memory |
| PC ops toolkit | [pc-tools\](pc-tools/) | - | - | SSH/deploy/bridge shortcuts for the board |

**Status (2026-11-10):** the **C version is production** — dashboard at
<http://192.168.1.22:8888/>, single gated `/stream` (annotated default,
`?raw=1` clean), capture suspended ~12 s after the last viewer closes,
inference fully native ORT (~1.2 s/forward). The python sections below
are kept in full because (a) they are the parity reference the port was
built against, (b) camkit is still the live tuning tool for the sensor.

---

## Contents

| § | topic |
|---|---|
| 0 | [Why the pipeline looks the way it does](#0-why-the-pipeline-looks-the-way-it-does) (hardware wall, measured) |
| 1 | [`frame_pumper.c` — the C frame copier](#pumper) (kernel ABI, PDEATHSIG, pipe discipline) |
| 2 | [`camkit.py` — the camera lab](#camkit) (backend ladder, phase verify, WB fix, endpoints) |
| 3 | [Bird system — python era](#birdpy) (what the python birdboard does; hand-over rules) |
| 4 | [Known limits / gotchas](#limits) |
| 5 | [birdcpp — the C port (production)](#cpp) (architecture, gating, native ORT, service, A/B, ORT probe) |
| 6 | [Bring-up timeline](#timeline) — how we got everything running on the board |


---

## 0. Why the pipeline looks the way it does

### The hardware wall (measured, not guessed)

| Path | Result |
|---|---|
| Sensor itself (kernel-side discard, zero copies) | **~28 fps** (sensor is real 30 fps) |
| Plain `os.read()` of a 4 MB raw frame (old bird path) | **~11 fps**, CPU ~125 % |
| Python mmap + single memcpy out of DMA memory | **11 fps**, fetch alone **53.6 ms** |
| Kernel pipe via `v4l2-ctl --stream-to=-` | **14.6 fps** |
| C `frame_pumper`, 1 thread | **14.1 fps** |
| C `frame_pumper`, 2 threads | **15.1 fps** |
| C `frame_pumper`, 4 threads | **21.4 fps** (delivered by the pipe) |

**Root cause:** the camera's DMA buffer (mmap'd or read) lives in memory
that the CPU reads **uncached** on this SoC — roughly 12 ms per MB,
i.e. ~60–80 MB/s. Every frame is 4.15 MB (1920×1080×2), so *any* path
that copies a frame into userspace is bounded by that bandwidth.
Parallelism works around it (uncached reads are latency-bound, so 4
independent memcpy streams multiply effective bandwidth), which is why
the C pumper with threads is the fast backend.

### WiFi is NOT the bottleneck
A view frame is a ~15 KB JPEG (≈0.5 MB/s at 30 fps). All bottlenecks are
on the board.

---

## 1. `frame_pumper.c` — the C frame copier {#pumper}

Files:
- Shared source in this repo: [`frame_pumper\frame_pumper.c`](frame_pumper/frame_pumper.c)
  (used by **both** bird versions and the camkit lab)
- Board binaries: `/home/root/camkit/frame_pumper`, also copied to
  `/home/root/bird/frame_pumper` and `/home/root/birdcpp/frame_pumper`.

### What it does

```
usage: frame_pumper /dev/video0 1920 1080 [threads] [frames]
```

1. Opens the V4L2 device `O_RDWR`.
2. `VIDIOC_REQBUFS count=4` → `QUERYBUF` → `mmap()` PROT_READ per buffer.
3. `QBUF` all buffers → `STREAMON`.
4. Loop: `poll()` → `DQBUF` → **copy the frame to a userspace buffer in
   N parallel `pthread` chunks** (frame split evenly, thread count from
   argv) → `write()` to stdout → `QBUF` back → next frame.
5. On exit: `STREAMOFF` + `REQBUFS count=0` (frees the vb2 queue).

### Points that cost real debugging time

- **Kernel ABI (5.10 TI vendor tree):** `v4l2_requestbuffers` is **20**
  bytes (mainline would be 24 or 28: `capabilities` u32 + `reserved[4]`),
  and `v4l2_buffer` is **88** bytes. The correct ioctl numbers seen by
  `strace`: REQBUFS `0xC0145608`, QUERYBUF `0xC0585609`, QBUF
  `0xC058560F`, DQBUF `0xC0585611`, STREAMON `0x40045612`, STREAMOFF
  `0x40045613`. The python side probes several REQBUFS sizes instead of
  hardcoding 20, so small kernel drift survives.
- **`PR_SET_PDEATHSIG` (SIGKILL) at startup:** the child never survives
  its parent. Without this, a camkit restart left a pumper child
  streaming forever while holding `/dev/video0` → every later `S_FMT`
  got `EBUSY` ("Device or resource busy") from Python. This was the
  single worst failure mode; the fix is one `prctl()` call.
- **Pipe writes must be a write-all loop.** A single
  `write(1, buf, 4 MB)` can come back short or `EINTR` (the pipe fills
  at 64 KB default/1 MB bumped); strace showed "write short" and the
  consumer received fewer bytes than a frame. The loop:
  ```c
  while (left > 0) {
      ssize_t n = write(1, p, left);
      if (n < 0) { if (errno == EINTR) continue; break; }
      p += n; left -= n;
  }
  ```
- `setvbuf(stdout, NULL, _IONBF, 0)` so partial frames are never held in
  libc buffering.
- Compilation on-board: `gcc -O2 -pthread -o frame_pumper frame_pumper.c`
  (gcc exists on the board image).

---

## 2. `camkit.py` — the camera lab (service `camkit.service`, port **8889**) {#camkit}

Purpose: live view + **every** driver control (exposure, gains, test
patterns, flips, modes, fps) to tune the camera. Lives at
`camkit\` (PC) ↔ `/home/root/camkit/` (board).

### Backend chain (tried in order, name shown in `/status`)

```
PipeCap (frame_pumper child, 4 threads)  ~15-20 fps view
  -> MMapCap (ctypes DQBUF loop)          ~11 fps view
     -> ReadCap (plain os.read)           ~10 fps
```

- **PipeCap** spawns `frame_pumper` with 4 threads and reads the pipe
  with **`os.readv` straight into a persistent `bytearray`** — the
  kernel copies into cached userspace memory, so Python pays **zero**
  per-frame copies (`del buf[:]`, `bytes(buf)` variants each cost a
  4 MB copy and measured slower). The pipe is widened to **1 MB** with
  `fcntl(rfd, 1031 /* F_SETPIPE_SZ */, 1<<20)`.
- **MMapCap** is pure ctypes: REQBUFS(4) → QUERYBUF → python `mmap` →
  QBUF → STREAMON → poll+DQBUF. One `bytes(mm[:npix*2])` memcpy (uncached
  reads are slow, but one sequential memcpy beats numpy views over mmap).
  On ANY failure it calls its own `stop()` — STREAMOFF, unmap all,
  REQBUFS(0) — because **a leaked vb2 queue makes every later S_FMT fail
  with EBUSY forever** on this shared-queue driver.
- **ReadCap** is the old plain `select`+`os.read` loop as last resort.

### Things that bite

1. **Open the device `O_RDWR`** — with `O_RDONLY`, `STREAMON` fails, the
   mmap cleanup leaves REQBUFS allocated, and every future `S_FMT`
   returns EBUSY.
2. **Teardown before reconfigure.** Flips (`V4L2_CID_HFLIP/VFLIP`) and
   mode changes are **ignored while the node streams** on this driver.
   camkit's control loop therefore: on any (re)configure job →
   `cap.stop()`, `os.close(fd)`, then full session rebuild. This fixed
   "flips never apply".
3. **One camera, one owner.** `camkit.service`:
   ```ini
   ExecStartPre=-/bin/systemctl stop birdboard
   ExecStartPre=-/bin/sh -c "/usr/bin/pkill -f frame_pumper 2>/dev/null; exit 0"
   ExecStopPost=-/bin/systemctl start birdboard
   ```
   (`systemctl` is at `/bin/systemctl` on this image.) When camkit stops,
   birdboard is restarted automatically and owns the camera again.
4. **EBUSY diagnosis recipe** (always means a queue owner exists):
   `pkill -f frame_pumper`, stop camkit, stop birdboard (ExecStopPost
   auto-starts it — stop it AGAIN right after), then reconfigure.
   Holders are visible via a `/proc/*/fd` readlink loop for
   `*/dev/video0*`.
5. **Warmup frames lie.** After STREAMON the vb2 ring drains old frames
   in a burst — avg-fps meters must skip the first ~3 frames, or you
   read nonsense like "291 fps".

### Bayer phase re-verification

At every open, camkit switches the sensor to the **solid-red test
pattern** (`test_pattern=2`, R=1023, G=B=0), debayers one frame with
each of the 4 OpenCV codes and picks the phase with the highest
`R / mean(B,G)` ratio. Needed because flips/mode changes can swap the
phase; wrong phase = wrong colors (blue birds!). Bird server currently
assumes `BayerBG2BGR` (verified once with the same pattern).

### fps presets

The driver has no fps control: frame period = `(H + vertical_blanking) ×
line_time`. Line time is **deterministic from kernel-reported values**:
`line_us = (width + horizontal_blanking) / pixel_rate` = **18.9035 µs**
at 1080p (1920+1528)/182.4 MHz. camkit computes `vblank = 1e6/(fps ×
line_us) − H` for any requested preset. (Exposure auto-clamps to
frame_height − 4.)

### White balance / green cast (the fix for the greenish feed)

Raw IMX219 without white balance reads **green-heavy** (the G sites'
response is the strongest and OpenCV's simple debayer adds no color
correction). camkit + bird both now apply:

1. **Gray-world AWB** — perm frame, sample the view at `::16` stride,
   `target = mean(G) / mean(channel)` clipped to 0.6–2.2, EMA-smoothed
   (0.05 weight) so there is no flicker.
2. **Composed uint8 LUT** — WB gains + optional shadow gamma
   (`v = i^0.55`) baked into ONE `(1,256,3)` LUT, applied with a single
   SIMD `cv2.LUT` call. Rebuilt only when gains drift (the earlier
   approach — float32 multiply over every pixel — cost ~10 ms/frame).
   This is what turns the "greenish" raw look into normal color.

In camkit both are toggles on the form + `/set` (`wb=1`, `gamma=1`);
WB is ON by default, gamma off by default.

### Endpoints

| EP | What |
|---|---|
| `/` | lab UI (live view + all controls) |
| `/mjpg` | multipart MJPEG live stream |
| `/frame.jpg` | latest frame |
| `/status` | JSON: fps, backend, phase, applied vs kernel-read-back values |
| `/set` | POST form → applies controls, redirects to `/` |
| `/readctl` | re-read controls from the kernel |

Note: `S["want"]`/`DEF_WANT` reset on every service restart (no
persistence by design).

---

## 3. Bird system integration (the same pipeline, in python `birdboard`) {#birdpy}

The python-era live system lives in [`python\`](python/) — from
`board_camera_server.py`
(runs as `birdboard.service` on **:8888**, board folder
`/home/root/bird/`):

- **Same backend chain** in `csi_session`: `PipeCap` → `MMapCap` →
  `ReadCap`; backend + live fps are reported in `/phone` JSON. Copy of
  the pumper binary at `/home/root/bird/frame_pumper`.
- `csi_session` opens the device `O_RDWR|O_NONBLOCK`, `csi_configure()`
  (media-ctl pad formats + `v4l2-ctl --set-fmt` + gain/exposure) stays
  as it was; the session finally-block calls `cap.stop()` — explicit,
  because PDEATHSIG only protects against process death, not session
  retries inside a live process.
- **No disk captures while the live feed runs** (user requirement):
  - `live_detect.sh`'s numbered `live_%d_det.jpg` rolling snapshots are
    retired (kill any `pkill -f live_detect.sh` leftovers).
  - `infer_loop` no longer writes `meta.json` / `frame_det.jpg` /
    `live_det.jpg`; metadata JSON is in-memory (`S["meta_txt"]`).
  - `board_detection.run(..., return_annotated=True)` returns the
    annotated JPEG in `meta["annotated_jpg"]` (results → re-encode only
    when boxes exist; no detections → the raw JPEG bytes pass through).
  - `/det.jpg` serves the in-memory annotated buffer on :8888.
- **White balance/gamma with the same composed-LUT trick as camkit**
  (bird had gray-world floats before; the LUT version does identical
  color with one `cv2.LUT` per frame). Exposed as dashboard toggles:
  `POST /camera-set` (`wb=1&gamma=1` body, checkbox semantics — absent
  key = off), state also visible in `/phone` JSON. Toggle changes take
  effect on the next captured frame; both default ON.
- Dashboard "Pages" card lists every endpoint; the camera panel polls
  `/frame.jpg` every 150 ms now (was 400 ms) to match the real capture
  rate — `/stream` remains the full-rate path.
- `/stream` upgraded from 1 fps throttle to near-capture-rate MJPEG.
- CPU governor pinned to `performance` at boot of the process (A53
  governor otherwise parks cores and halves the pipeline).
- DroidCam phone path untouched (`camera_url.txt` = `http://...` uses it;
  `/dev/video0` uses CSI).

### Hand-over rules (unchanged)

| Task | Command |
|---|---|
| Camera lab | `systemctl start camkit` |
| Return to bird | `systemctl stop camkit` (birdboard auto-starts) |
| Camera-free tests with no camera owner | `systemctl stop camkit; systemctl stop birdboard` |

Board IP on the home network: **192.168.1.22** (`tibird` alias), WPA
config restored to `/etc/wpa_supplicant.conf.orig` (Saumya_5G /
`9869036104airtel`), bring-up via `/etc/wifi-up.sh`.

---

## 4. Known limits / gotchas {#limits}

- **View fps ceiling ≈ 15 on this board** (pumper 21 is *pipe
  delivery only*; add python debayer/encode and the wall lands ~15).
  Sensor = 30. Anyone promising more is wrong.
- `1640x1232` mode: horizontal_blanking changes (1808 vs 1528), so
  line_us stays 18.9035 µs — presets still exact.
- REQBUFS struct-size probe list exists because the 5.10 vendor ABI is
  nonstandard; never "clean it up" to mainline sizes without testing.
- camkit resets its `want` state on restart, and now **boots with the
  user-tuned defaults**: mode `1640x1232` (2x2 binned), exposure 1759,
  analogue_gain 200, digital_gain 4095, vblank 683 (~27.6 fps sensor
  rate), view half (820x616), JPEG quality 80, white balance ON,
  shadow lift ON, no flips, test pattern 0.
- birdboard's `csi_configure` applies the same tuned values from
  constants (`CSI_GAIN=200`, `CSI_DIG_GAIN=4095`, `CSI_EXPOSURE=1759`,
  `-c vertical_blanking=683`, pads/fmt at 1640x1232, preview 820x616).
  Verified live: both report `want == ctl` (kernel read-back) for every
  control.
- Exposure range quirk: the driver's max exposure is
  `(frame_height + vertical_blanking) - 4`. In 1080p + vblank 683 the
  range was 4..1759 (1759 = the max); in 1640x1232 + vblank 683 it is
  4..1911, so the tuned 1759 applies un-clamped.
- Old `live_*_det.jpg` files in `python\artifacts\` are historical
  output of the retired `live_detect.sh` loop; new captures never appear.

---

## 5. birdcpp — the C port (production) {#cpp}

Everything above describes the python birdboard. [`cpp\`](cpp/) in this repo
holds a full C99 port of it (`birdcpp.c`, single file, ~2,800 lines)
that keeps every route, the dashboard HTML, the /camera-set semantics
(checkbox absent = off, scalars absent = unchanged, fps preset
overrides vblank via measured line_us, mode/flip change tears down and
rebuilds the session) and the user-tuned defaults already documented
in section 5. Same camera chain, same `frame_pumper` helper.

**Design change (live-media-only, as of 2026-11-10)**: the snapshot
JPEG endpoints are gone - `/frame.jpg` and friends answer **410**. The
camera runs only while someone watches `/stream`: every stream viewer
registers as a consumer, and a gate thread SIGSTOPs the pumper 12 s
after the *last viewer event* (open or close); a new viewer SIGCONTs
it instantly. All state stays in RAM - nothing is written to disk while
streaming. Inference moved fully into C (native ORT below); no python
runs while birdcpp is up.

### Architecture (per frame, all C)

    frame_pumper (fork/exec, PDEATHSIG, same binary)
      -> 1 MB-wide nonblocking pipe (F_SETPIPE_SZ; per-frame progress
         kept across poll timeouts so a 4 MB frame never realigns)
    -> phase verify per session: solid-red test pattern scored per
       lattice site (2 frames), restores want values afterwards
    -> debayer 4-phase: 2x2 Bayer bin for the "half" view, threads
       (row-split) bilinear for "3/4" (bin + bilinear upsample) and
       "full" (full bilinear)
    -> gray-world AWB EMA + gamma 0.55 LUT (python formulas, table
       lookups only in the hot loop)
    -> libturbojpeg (-l:libturbojpeg.so.0, API declared locally: board
       ships no turbojpeg.h); TJPF_BGR verified empirically with
       `--tjtest` (solid red -> decodes R=254, B=0)
    -> native ORT forward (640x640 letterbox, cv2-parity) -> YOLOv8
       NMS (conf 0.25 / IoU 0.45) -> boxes drawn with a builtin 5x7
       font -> meta JSON -> annotated frames feed the stream
    -> HTTP: / (single live card) /stream?raw=1 /meta.json /phone
       /camera-set (POST + GET, 303) /ping; /frame.jpg /det.jpg
       /snapshot.jpg answer 410; every jpg served is an owned copy
       (no use-after-free on swap)

### Capture gating (idle = no camera work)

`/stream` viewers increment `stream_live` (+`stream_raw` for `?raw=1`);
open marks activity, close refreshes the activity timestamp. The gate
thread wakes every 2 s: `stream_live > 0` -> SIGCONT; `stream_live == 0
&& now - activity_t > 12 s` -> SIGSTOP, journal line
`pumper suspended (idle 12 s)`. Resume is instant
(`pumper resumed (stream opened)`) and the ORT session, sensor config
and phase verify all stay warm - no re-create on reopen. **Only**
`/stream` (i.e. the dashboard's embedded `<img src="/stream">`) starts
capture; `/phone`, `/ping`, `/meta.json`, `/` and `/camera-set` accept
activity but never start it. The pumper's pid is published in the
state struct; the capture loop skips encoding entirely while suspended
and reaps a dead child before re-enabling.

### Inference is fully native (no python anywhere)

onnxruntime 1.31.1 (official `onnxruntime-linux-aarch64` C API
package) at `/home/root/ort/`, **dlopen'd at runtime**
(`/home/root/ort/lib/libonnxruntime.so`; the board's libstdc++ takes
no link-time bind, see the probe notes below). Per detection cycle:

    letterbox 820x616 view -> 640x640 f32 CHW canvas, 114-gray pad,
    paste 481x640 @79,0 (cv2-parity geometry, ~60 ms)
    -> OrtValue over the caller's buffer -> api->Run (4 intra threads)
    -> parse 1x5x8400 -> class score >= 0.25, de-letterbox boxes,
       xywh -> xyxy, NMS IoU 0.45 (max 64 dets)
    -> draw   (5x7 font felt-tip boxes + label) -> publish annotated
       JPEG + meta JSON `{"file":"frame.jpg","n":..,"infer_ms":..,
       "size":[vw,vh],"dets":[...]}` - keys parity with the python era

`infer_thread` runs it only while a stream consumer exists; results
older than `DET_IDLE_S` 15 s draw as clean frames. `--orttest`
self-tests: synthetic frame -> letterbox -> two forwards, prints both
meta JSONs (cold ~3 s, warm ~1.16 s quiesced) and fails nonzero on
error. `--ortlib PATH` and `--ort-threads N` (default 4) override.

### Reality check on speed

The view pipeline lands ~15-19 fps - the same wall as python,
because 4 MB/frame DMA copy + 640-px YOLOv8 forward dominate, not the
interpreter. The wins: per-frame python overhead is gone (capture,
debayer, color, JPEG, HTTP all native), memory drops, and the server
survives bursts without GC pauses. Detection: **~1158 ms steady**
(quiesced, 4 threads) vs ~2.3 s under cv2.dnn and ~2.5-2.8 s for the
first forward when the live co-run inflates it; ORT session create
~350-660 ms once, then warm for the process lifetime. And with the
gate, all of that is **zero** when nobody is watching.

### Service

`/etc/systemd/system/birdcpp.service`:

| Task | Command |
|---|---|
| Run the C birdboard | `systemctl start birdcpp` (port 8888; stops birdboard) |
| Return to python birdboard | `systemctl stop birdcpp`, then `systemctl start birdboard` (NO auto-restart - gated design has no continuous capture to hand back) |
| Manual C test instance | separate ssh calls: stop both, `pkill -f fram[e]_pumper`, then `cd /home/root/birdcpp && setsid nohup ./birdcpp -p 8890 &` (pkill hazard: never in a call that also contains the literal target text) |

`camera_url.txt` at `/home/root/birdcpp/camera_url.txt` (= `/dev/video0`).

Status verified live 2026-11-10 (service active on 8888): /ping OK,
/frame.jpg + /det.jpg -> 410 with JSON explanation, /stream 200 with
real MJPEG, /phone LIVE fps 12.3 with detection meta
(`infer_ms` 2543 first-live-run), journal shows
`pumper suspended (idle 12 s)` exactly 12 s after viewer close and
instant resume with **no second ORT session** (warm context). Settings
roundtrip suite historically 19/19 PASS (exposure / gain / vblank /
fpreset math 531 exact / quality / pattern+pixels / scale with
1230x924 preview dims / mode + hflip rebuilds / wb+gamma); the suite
predates the endpoint removal, see README-CPP.md notes.

### Gotchas for the C side

- `pkill -f birdcpp` while a remote shell's own cmdline contains the
  literal kills that shell - and **bracket patterns do not save you
  when the target literal appears elsewhere in the same cmdline**
  (`... setsid nohup ./birdcpp -p 8890 ... && pkill -f 'bir[d]cpp'`
  is one command line for the remote shell; its own argv carries the
  literal and pkill matches it). Rule: keep pkill inside board-side
  script files, or in ssh calls that do not embed the target text.
- PC-side curl caveats: `curl -o 'C:\Users\Jay singh\...'` breaks on
  the space in the user dir (use `-o NUL`, plain stdout, or run the
  curl on the board); PowerShell interpolates `$(...)` / `$_` inside
  double-quoted ssh args; scp destinations from vars need `${B}:path`.
- Phase naming: birdcpp reports the raw lattice directly (`RG` =
  top-left R = IMX219 native RGGB). The python era's "phase BG"
  string was that code's cvtColor decode-arg name - same sensor,
  same mapping, different label. Do not "fix" one to match the other;
  the solid-red roundtrip through the real pipeline is the truth
  (verify via a raw-view stream grab; `/frame.jpg` no longer exists).
- Annotated `/stream` 200-with-zero-bytes was the boot-placeholder
  bug (python-era `view_bgr` pre-alloc without `det_bgr`) - fixed
  2026-11-10 with a boot-time det canvas + self-healing
  `encode_jpg_det`; diagnostics live in `/phone`
  (`stream_live/stream_raw/det_pubs/raw_pubs`) and the journal
  `gate:` / `bufs-ok:` lines.

### Head-to-head A/B: python birdboard vs birdcpp (2026-11-09)

45 s windows, governor `performance`, identical tuned defaults
(`bench_sample.py` on the board):

| metric                   | python birdboard | birdcpp (C)              |
|--------------------------|------------------|--------------------------|
| view fps (/phone)        | 9.98             | **15.18**                |
| system CPU               | 88.1 %           | 84.7 %                   |
| per-proc CPU             | server 186.6 %   | 36 + 111 + 106 (3 procs) |
| RSS total                | 187 MB           | 173 MB                   |
| /frame.jpg median (p95)  | 22.8 (40.7) ms   | **13.0 (25.9) ms**       |
| / (root) median          | 23.3 ms          | 10.8 ms                  |
| infer_ms per det frame   | 2245-2448        | 2229-2380                |

The 1.5x pipeline win comes from the C hot loop. In this table's era
the detection wall (~2.3 s) was the YOLOv8 forward inside `cv2.dnn`.
**As of 2026-11-10** that wall is native ORT: `infer_ms` steady
~1158 ms quiesced, and the `/frame.jpg` rows are historical (endpoint
removed, live-media-only design; the dashboard streams `/stream`).

### Native ORT on the board (feasibility probe -> shipped)

Official `onnxruntime-linux-aarch64-1.31.1` (C API package) at
`/home/root/ort/`: `ldd` resolves on glibc 2.30 (needs <= 2.28; max
GLIBCXX 3.4.22 satisfied by TI GCC 9.2). The board `libstdc++.so.6`
has no symbol-version records, so GNU ld **refuses the link-time bind**
(-lonnxruntime fails on `@GLIBCXX` versioned refs) - the working route
is runtime binding, `dlopen` + `dlsym` of `OrtGetApiBase` (the same
load model cv2 itself uses). `ort_probe.c` on the board, camera
quiesced:

| config                 | session create | steady forward |
|------------------------|----------------|----------------|
| ORT v1.31.1 intra=4 th | 235 ms         | **1123 ms**    |
| ORT intra=2 th         | -              | 1544 ms        |
| cv2.dnn (python-era)   | -              | ~2300 ms       |

Model IO confirmed through the C API: `images` 1x3x640x640 -> `output0`
1x5x8400 (4 box + 1 class, YOLOv8 head). The probe's projection was
**shipped 2026-11-10**: birdcpp.c now does letterbox + NMS + annotate +
JSON in C with dlopen ORT (see "Inference is fully native" above) -
detection wall halved to ~1158 ms quiesced, python sidecar removed
from the board, 4 intra threads confirmed optimal on the 4xA53.
`ort_probe.c` stays in [`cpp\`](cpp/) as the frozen reference
for the dlopen/dlsym pattern.

---

## 6. Bring-up timeline — how we got everything running {#timeline}

Condensed history of the board bring-up, worst traps included, in the
order they actually happened:

1. **Raw sensor path** — first contact with the IMX219 over CSI:
   `media-ctl`/`v4l2-ctl` pad configuration, 10-bit RG10 unpacking, the
   4.15 MB/frame DMA-copy wall (§0/§1). Lesson: measure every candidate
   path; the obvious ones lose.
2. **frame_pumper** — the C frame copier: kernel-ABI probing against the
   5.10 TI vendor tree, 4-thread uncached memcpy ladder, PDEATHSIG to
   stop `$EBUSY` orphans, write-all pipe discipline. This became the
   single capture stage shared by every later component (§1).
3. **camkit** — the tuning lab: test-pattern lattice scoring to pin
   Bayer phase, gray-world AWB + gamma LUT to kill the green cast, all
   scalar driver controls exposed on :8889. This is where the camera
   got *usable* pictures (§2).
4. **python birdboard** — the first end-to-end bird system: same
   pumper + phase verify, in-process `board_detection` YOLOv8, live
   dashboard on :8888, no-disk policy for detection artifacts, user-tuned
   sensor defaults (§3). This is the system the C port was verified
   against route-by-route.
5. **C port (birdcpp)** — same routes and semantics in single-file C99,
   turboJPEG locally declared, then the two design leaps:
   - **live-media-only + capture gating**: snapshot JPEGs deleted (410),
     `/stream` is the only capture trigger, pumper SIGSTOPped 12 s after
     the last viewer, instantly resumed on the next viewer, ORT session
     kept warm (§5).
   - **native inference**: onnxruntime 1.31.1 C API via dlopen (the
     board's libstdc++ refuses a link-time bind), cv2-parity letterbox,
     NMS, 5x7-font annotation, meta JSON - python fully gone from the
     board (§5).
6. **Wrap-up lessons** (all in the cpp README troubleshooting):
   the annotated-canvas placeholder bug that blank the default stream
   while raw still worked (fixed + self-healing encoder), two service
   restarts that silently re-detected Bayer phase (`RG` ↔ `GB` labels),
   the pkill-vs-ssh self-kill footgun, and curl/PowerShell quoting
   traps. If it can bite, it is written down either here or in
   [`cpp\README.md`](cpp/README.md).
