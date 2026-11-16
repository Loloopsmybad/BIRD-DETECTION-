# birdcpp - the C99 birdboard (copy of the python birdboard project)

A port of `Bird Live System/board_camera_server.py` to a single-file
C99 program for the TI SK-AM62 (Arducam IMX219 CSI). Same dashboard,
same `/camera-set` semantics, same user-tuned defaults. **Live media
only**: there are no snapshot JPEG endpoints anymore - the camera runs
only while someone watches `/stream`, and the pumper is SIGSTOP-ed
the rest of the time. Inference is fully native: onnxruntime 1.31.1
C API, loaded with dlopen at runtime (the board's libstdc++ cannot
take a link-time bind); letterbox, YOLOv8 NMS, box drawing (builtin
5x7 font) and the result JSON are all C. No python survives.

    camera -> frame_pumper (C, fork/exec, pipe)
           -> birdcpp: pipe in / phase verify / debayer / AWB+gamma /
              turbojpeg / own HTTP server (MJPEG /stream, gated)
           -> same process: native ORT forward, NMS + annotate, meta
              JSON; the annotated frames feed the stream and /meta.json

`reference\` is a frozen snapshot of the python project this was
ported from (including the port-era CAMERA_PIPELINE.md and the retired
`bird_infer_sidecar.py`). The service unit lives in `deploy\`.

## Files

| file | role |
|---|---|
| `birdcpp.c` | the whole server: capture, debayer, color, JPEG, HTTP, settings, native ORT inference, capture gating |
| `ort_probe.c` | a frozen feasibility probe for the runtime-dlopen ORT route (kept as reference) |
| `Makefile` | `gcc -O2 -std=gnu99 -Wall -pthread -I/home/root/ort/include ... -l:libturbojpeg.so.0 -lm -ldl` |
| `deploy\birdcpp.service` | systemd unit (stops birdboard on start; NO auto-restart of it on stop) |
| `reference\Bird Live System\` | python parity source |

Board deployment tree (also the runtime paths baked into the C):

    /home/root/birdcpp/birdcpp            binary + birdcpp.c
    /home/root/birdcpp/best.onnx          (copied from /home/root/bird)
    /home/root/birdcpp/frame_pumper       (copied from /home/root/bird)
    /home/root/birdcpp/camera_url.txt     (= /dev/video0)
    /home/root/ort/include + lib/         onnxruntime-linux-aarch64-1.31.1

## Build

On the board (the ORT headers must be present; `-ldl` for dlopen):

    cd /home/root/birdcpp && make

First-time sanity: `./birdcpp --tjtest` writes `/tmp/tjtest_red.jpg`
and `blue`; decode with cv2 - red must decode red / not blue (proves
the locally-declared TJPF_BGR numbers; measured R=254). `./birdcpp
--orttest` runs the inference self-test: synthetic 820x616 frame ->
cv2-parity letterbox (640x640 canvas, 114 pad, paste 481x640 @79,0) ->
two printed forward roundtrips (cold, then warm) -> returns nonzero
on failure.

## Deploy (from the PC, ssh alias shown short: `-o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa root@192.168.1.22`)

    scp birdcpp.c Makefile root@IP:/home/root/birdcpp/
    ssh root@IP "cd /home/root/birdcpp && make"
    ssh root@IP "cp /home/root/bird/best.onnx /home/root/birdcpp/ 2>/dev/null;
                 cp /home/root/bird/frame_pumper /home/root/birdcpp/ 2>/dev/null"

`/home/root/ort` (include + lib) is deployed once and left alone; if
it moves, adjust the `-I` in the Makefile and `ORT_LIB_DEFAULT` /
`--ortlib` in birdcpp.c.

## Run

Normal (systemd, port 8888, replaces python birdboard; birdboard is
NOT auto-restarted on stop anymore - the live-media design has no
continuous capture to hand back, start birdboard manually if wanted):

    systemctl start birdcpp      # stops birdboard via ExecStartPre
    systemctl stop birdcpp       # camera goes quiet; nothing auto-started

Manual test instance (no disk writes anywhere; logs to /tmp only).
Three **separate** ssh calls - combining them in one command puts the
pattern and the literal `birdcpp` in the same remote cmdline and
pkill kills that shell itself (see troubleshooting):

    ssh root@IP "systemctl stop birdboard birdcpp 2>/dev/null; true"
    ssh root@IP "pkill -f fram[e]_pumper 2>/dev/null; true"
    ssh root@IP "cd /home/root/birdcpp && setsid nohup ./birdcpp -p 8890 >/tmp/birdcpp.log 2>&1 &"

(to kill the test instance later, a bracketed pattern is fine **only**
in a call that contains no plain `birdcpp` text otherwise: ssh
root@IP "pkill -f 'bir[d]c[p]p -p 8890'").

Options: `-p PORT` (default 8888), `-m /path/best.onnx`, `--tjtest`,
`--orttest` (self-test, then exit), `--ortlib PATH` (override
`/home/root/ort/lib/libonnxruntime.so`), `--ort-threads N` (default 4).

## Endpoints

`/` dashboard (single live card with the embedded annotated stream),
`/stream` multipart MJPEG - **annotated default, `?raw=1` for the
clean view** - `/meta.json`, `/phone` (same keys incl.
want/ctl/line_us/preview), `/camera-set` POST+GET -> 303, `/ping`.

### Model toggle (dashboard, since 2026-11-16)

`/model` GET -> JSON: `{current, current_path, img, switching, err,
models:[{name, path, bytes, current?}]}` — `models` is every
`*.onnx` in `/home/root/birdcpp` + `/home/root/birdcpp-scratch`
(sorted by name).

`/model-set?model=X` GET (preferred; the POST form field caps at 23
chars) -> `{ok, switching_to}` and queues the switch, which the
infer thread executes between cycles: ORT session rebuild (~1-2 s,
detections pause briefly), letterbox **auto-adapts** to the model's
static input size (square, multiple of 32, 320..640; anything else
fails the load), then detections resume. `X` is a basename (searched
in both model dirs) or an absolute `.onnx` path. Bad targets answer
`400 {ok:false, err:...}`; a failed load rolls back to the previous
model and surfaces the error on `/model` `err` + dashboard.

The choice persists per port to `/home/root/birdcpp/model_sel_<port>.txt`
(wins over the unit `-m` on restart; 8888 and 8890 are independent so
the sandbox can never move production). Delete the file to return to
the unit default. `/phone` gained `"model": {name, img, switching}`.

Gone (the point of the design): `/frame.jpg`, `/det.jpg`,
`/live_det.jpg`, `/snapshot.jpg` and friends answer **410** with a
one-line JSON explanation. Capture runs only while a `/stream`
consumer exists; the last viewer closing schedules a SIGSTOP of the
pumper **12 s later** (a new viewer SIGCONTs it instantly, with the ORT
session, sensor config and phase verify all kept warm - the journal
lines read `pumper suspended (idle 12 s)` / `pumper resumed (stream
opened)`). `/phone`, `/ping`, `/meta.json` and the dashboard HTML
itself never start the camera - only the embedded `/stream` <img> does.

`/camera-set` semantics (verbatim from python `do_POST`): checkbox
absent = off (wb / gamma / hflip / vflip), scalar absent = unchanged,
`fpreset>0` overrides vblank via the measured line_us, mode/flip
changes set the reconf flag (session rebuild, instant re-entry), live
scalar apply only when the session survives. Phase verification reruns
per session (solid-red lattice scoring) and restores the want values.

## Self-check / roundtrip suite

`verify8890.py` (built from the python-era suite; HTML check matches
the real `<select name="mode">` markup this time). Run against a
live instance on 8890: expects the sensor defaults and reports
19/19 PASS when the server matches python semantics.

## Self-check / roundtrip suite

`verify8890.py` (built from the python-era suite; HTML check matches
the real `<select name="mode">` markup this time). Run against a
live instance on 8890: expects the sensor defaults and reports
19/19 PASS when the server matches python semantics.

**Note - predates the live-media-only rework**: some of its checks hit
`/frame.jpg` and `/det.jpg`, which now intentionally answer 410 - so
the suite will report failures on exactly those, and that is *expected*,
not a regression. The semantics it really probes (`/camera-set`,
`/phone`, phase labels, tuning defaults) still pass. Re-run it only
when touching the capture/config path, and mentally exclude the
snapshot-endpoint checks (or prune them from the script).

## Benchmark: python birdboard vs C birdcpp

45 s sampling windows (governor `performance`, identical tuned defaults,
`bench_sample.py` run on the board, phase verify excluded):

| metric                  | python birdboard | birdcpp (C)             |
|-------------------------|------------------|-------------------------|
| view fps (/phone)       | 9.98             | **15.18**               |
| system CPU              | 88.1 %           | 84.7 %                  |
| per-proc CPU            | server 186.6 %   | 36 + 111 + 106 (3 procs)|
| RSS total               | 187 MB           | 173 MB                  |
| /frame.jpg median       | 22.8 ms          | **13.0 ms** (p95 25.9)  |
| / (root) median         | 23.3 ms          | 10.8 ms                 |
| infer_ms (per det frame)| 2245-2448        | 2229-2380               |

The pipeline win is 1.5x. In this table's era the detection wall
(~2.3 s) was identical in both: it was the model itself inside
`cv2.dnn`.

**Note - as of the live-media-only rework this table is partly
historical**: detection moved fully into C (native ORT, see the next
section), so `infer_ms` is no longer `2229-2380` but **~1158 ms
quiesced**, ~2.5-2.8 s live in the original 4+4-thread configuration
(first forward of a session, thread pools cold) and **~1.7 s live
after the core partition (opt-B, below)**. The `/frame.jpg median` row
is stale - that endpoint does not exist anymore (live media only). The
view-fps / CPU / RSS rows remain valid measurements of the capture
pipeline itself.

## Detection stage timings (2026-11-10 instrumentation)

Every detection cycle carries per-stage `clock_gettime(CLOCK_MONOTONIC)`
timers: a one-line report every ~2 s in the journal
(`det-stage: snap .. letterbox .. fwd .. parse .. nms .. | cycle ..`)
plus a live `stage` block in `/phone`. Measured LIVE (annotated stream
open, one `/dev/video0` consumer):

| stage (function) | what | live |
|---|---|---|
| snap (`infer_thread` memcpy) | view frame -> infer snapshot | ~2-3 ms |
| letterbox (`ort_letterbox`) | bilinear resize + CHW float | ~60-100 ms |
| **fwd (`ort_run` -> ORT Run)** | **YOLOv8-nano 1x3x640x640 on 4xA53** | **see table below** |
| parse (confidence filter, 8400 cands) | - | ~0 ms |
| nms (`nms_pick` + box mapping) | - | ~0 ms |

The forward pass alone is therefore **~97% of the detection wall**;
nothing else in the port's code is slow.

Core partition experiment (live `fwd`, same camera, same viewer count):

| config | fwd avg | cycle | verdict |
|---|---|---|---|
| A: pumper 4 threads + birdcpp all cores free (original) | ~2.74 s | ~2.83 s | the starting point |
| **B: pumper pinned to core 3, 1 thread, nice 10 (shipped)** | **~1.74 s** | **~1.81 s** | **~36% more detections/s** |
| C: B + whole birdcpp process pinned to cores 0-2 | ~2.75 s | ~2.8 s | **rejected** - ORT's intra threads lose more from seeing only 3 cpus than they gain from the parked copier |

Shipped configuration: `pumper_spawn()` pins the copier child to
`PUMPER_CORE` (=3) with `sched_setaffinity`, sets priority +10 and
spawns it with `threads=1` (the 1-thread pipe still yields ~14 fps >=
the ~12 fps core-contended rate, so capture loses nothing).
`birdcpp` itself is deliberately NOT affinity-restricted (see C).
Detection cadence is now ~0.55/s (one annotated frame every ~1.8 s)
instead of ~0.35/s. The remaining lever is the model's imgsz
(640 -> 448/384 re-export), not any code in this port.

**Promotion (2026-11-10, later the same day)**: the `--img` runtime
flag was folded into production `birdcpp.c` (default still 640), the
448 model deployed as `/home/root/birdcpp/best_448.onnx`, and the unit
now runs
`ExecStart=/home/root/birdcpp/birdcpp -p 8888
            -m /home/root/birdcpp/best_448.onnx --img 448`
Accuracy was validated first (yolo val, 191 val images / 413 birds):
mAP50 0.832 (640) -> 0.806 (448), recall 0.753 -> 0.741 - a ~3% tax.
Verified production timings now: **fwd ~875 ms, cycle ~915 ms,
letterbox ~30 ms, capture ~14 fps** - about **3.1x the detection rate
of this morning's original configuration** (2830 ms cycle).
The 640 export stays on the board as `best.onnx`: instant revert is a
one-line ExecStart edit (see `deploy\birdcpp.service`), and the full
sandbox/methodology lives in `trial-a\` + `trial-a\README.txt`.

## Native ORT probe (inference fully in C?)

`/home/root/ort/` holds the official
`onnxruntime-linux-aarch64-1.31.1` package (C API, 10 MB). `ldd` fully
resolves it on this image (glibc 2.30, TI GCC 9.2): ORT needs max
GLIBC 2.28 and GLIBCXX 3.4.22 - both satisfied. The board's
`libstdc++.so.6` carries **no symbol-version records**, so strict GNU
ld **refuses a link-time bind** of libonnxruntime (`-lonnxruntime`
fails with versioned-`@GLIBCXX` undefined refs); the working route -
and the same one cv2 uses - is **runtime binding**: `dlopen` + `dlsym`
the exported `OrtGetApiBase`. Gotcha found on the way: `dlsym` returns
the *function* `const OrtApiBase *(*)(void)`, not the struct itself
(prototype it and call it; see `ort_probe.c`).

`ort_probe` timings on the board (camera quiesced, best.onnx
1x3x640x640 -> 1x5x8400, floating-point 4xA53):

| config                | session create | steady forward |
|-----------------------|----------------|----------------|
| ORT intra=4 threads   | 235 ms         | **1123 ms**    |
| ORT intra=2 threads   | -              | 1544 ms        |
| cv2.dnn (sidecar/both)| -              | ~2300 ms       |

Verdict: proved and then **shipped**. The C port is live in birdcpp.c:
letterbox is cv2-parity (640x640 canvas, 114-gray pad, paste
481x640 @79,0 for 820x616), NMS at conf 0.25 / IoU 0.45, boxes drawn
with a builtin 5x7 font, meta JSON identical to the sidecar's keys.
Measured on the live binary with the board quiesced: session create
~350-660 ms, steady forward **1158 ms @ 4 intra threads** (matches the
probe; the python birdboard on the other cores inflates any measurement
run next to it by ~2.5x - quiesce before believing numbers). The
python sidecar is deleted from the board; `--orttest` self-tests the
whole chain.

## Tuning constants (top of birdcpp.c)

`DEF_*` sensor defaults (exp 1759 / gain 200 / dig 4095 / vblank 683 /
pixels 1023), `MODE_WH` + `MODE_NM` sensor modes, `SCALE_*` view
factors, gamma 0.55 and the AWB EMA 0.95/0.05 - all mirror the python
constants; touch both copies together or they drift.

## Troubleshooting history (learned the hard way)

- **"Invalid handle" from tjCompress2**: tjInitCompress was only
  called in main() after the `--tjtest` early-return; the tjtest path
  now inits its own handle.
- **`pkill -f <pattern>` suicide over ssh**: the remote shell's own
  cmdline can contain the pattern - and **bracket patterns do not save
  you when the target literal appears elsewhere in the same cmdline**
  (proved twice: `ssh` running `... setsid nohup ./birdcpp -p 8890 ...
  && pkill -f 'bir[d]cpp' ...` kills the shell itself at the pkill,
  because the shell's own argv contains the literal `birdcpp -p 8890`).
  Rule: put pkill inside board-side script files (scp + `sh script.sh`),
  or use separate ssh calls that do not embed the target text.
- **PC-side background curl `-o` with a spaced path**: the user dir is
  `C:\Users\Jay singh` - a PowerShell `Start-Job` running
  `curl.exe -o 'C:\Users\Jay singh\...'` fails to write and the job
  reports success anyway (empty file / no file). Workarounds: run the
  stream curls **on the board** (`curl -o /tmp/x.bin` board-side), or
  foreground with `-o NUL`, or plain stdout. Also mind PowerShell
  interpolating `$(...)` / `$_` inside double-quoted ssh args and the
  `${B}:path` braces needed for scp destinations built from a variable.
- **Annotated stream empty (200 OK, zero bytes) while `?raw=1` worked** -
  the root cause of the 2026-11-10 "where is my live feed?" report.
  `main()` still carried the python-era placeholder: it pre-allocated
  `view_bgr` (gray boot view for the old `/frame.jpg`) and set
  `vw_alloc/vh_alloc` but NOT `det_bgr`. That made
  `ensure_view_bufs_locked` early-return every frame, so the annotated
  canvas was never lazily allocated either; `encode_jpg_det` NULL'd on
  every frame and the default `/stream` delivered nothing, while meta
  JSON, `/phone` fps and raw kept looking healthy and masked it.
  Fixed two ways: `det_bgr` is allocated at boot alongside the view,
  and `encode_jpg_det` self-heals (re-allocs on any future dim
  mismatch instead of silently returning NULL). Field diagnostics
  added: `/phone` now carries `stream_live`/`stream_raw`/
  `det_pubs`/`raw_pubs`, and the journal has rate-limited `gate:`,
  `bufs-ok:`/`bufs-alloc:` and `det canvas allocated` lines.
- **"Suspiciously slow" orttest numbers**: stopping the birdcpp service
  used to re-start the python birdboard via the old unit's
  ExecStopPost, which then hogged all four cores and inflated every
  forward timing ~2.5x. Quiesce (`pgrep -af bir|frame_pumper`) before
  believing any timing; governor + `scaling_cur_freq` next.
- **`nohup ... &` over ssh dies on disconnect**: use
  `setsid nohup ... </dev/null &`.
- **Phase label differs from the python era**: birdcpp reports the raw
  lattice directly (`RG` = top-left R = IMX219 native RGGB); the
  python-era "phase BG" string was that code's cvtColor decode-arg
  name - same sensor, same mapping, different label. The solid-red
  roundtrip through the real pipeline is the truth: pattern red ->
  the live stream decodes R=254, G=0, B=0 (verify with a raw-view
  grab + cv2; `/frame.jpg` no longer exists). Do not "fix" one label
  to match the other.
- **Annotated frames with nothing detected**: `process_frame`
  re-encodes the plain view (q82) in RAM so the stream keeps flowing
  when the last detection is empty - only frames with drawn boxes are
  authoritative detections. The stale-result timeout is
  `DET_IDLE_S` 15: after it, the stream shows the clean view even if
  the last forward had boxes.
