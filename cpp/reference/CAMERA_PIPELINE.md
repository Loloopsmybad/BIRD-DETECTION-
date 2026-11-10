# Camera Pipeline — Technical Documentation (frame_pumper + camkit)

Technical reference for the **CSI camera pipeline** shared between the
**camkit** camera lab (`D:\samsung\camkit\`) and the **bird detection**
system (`D:\samsung\BIRD-DETECTION\Bird Live System\board_camera_server.py`).

Everything runs on the **TI SK-AM62** board (4× Cortex-A53, kernel 5.10
vendor tree) with the **Arducam V2.2 (IMX219)** CSI camera on
`/dev/video0` (RG10, 2 bytes/pixel raw Bayer).

---

## 1. Why the pipeline looks the way it does

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

## 2. `frame_pumper.c` — the C frame copier

Files:
- PC source: `C:\Users\Jay singh\AppData\Local\Temp\opencode\frame_pumper.c`
  (mirror it into `D:\samsung\camkit\frame_pumper.c` if you want it kept)
- Board binary: `/home/root/camkit/frame_pumper`, also copied to
  `/home/root/bird/frame_pumper` for the bird system.

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

## 3. `camkit.py` — the camera lab (service `camkit.service`, port **8889**)

Purpose: live view + **every** driver control (exposure, gains, test
patterns, flips, modes, fps) to tune the camera. Lives at
`D:\samsung\camkit\` (PC) ↔ `/home/root/camkit/` (board).

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

## 4. Bird system integration (the same pipeline, in `birdboard`)

`D:\samsung\BIRD-DETECTION\Bird Live System\board_camera_server.py`
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

## 5. Known limits / gotchas

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
- Old `live_*_det.jpg` files in `Bird Live System/` are historical
  output of the retired `live_detect.sh` loop; new captures never appear.
