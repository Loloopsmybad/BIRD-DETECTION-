import http.client, time

c = http.client.HTTPConnection("192.168.1.3", 4747, timeout=8)
c.request("GET", "/video", headers={"User-Agent": "board-bird/1.0"})
r = c.getresponse()
print("STATUS", r.status, "CONTENT-TYPE:", r.getheader("Content-Type"))
c.sock.settimeout(20)

t0 = time.time()
total = 0
frames = 0
buf = b""
nreads = 0
read_sizes = []
last_frame = 0.0
snap = 0.0
while (time.time() - t0) < 25:
    try:
        d = r.read(65536)
    except Exception as e:
        print("READ_EXC", repr(e)[:60], "at t=%.1f" % (time.time() - t0))
        break
    if not d:
        print("EOF at t=%.1f" % (time.time() - t0))
        break
    nreads += 1
    read_sizes.append(len(d))
    total += len(d)
    buf += d
    while True:
        i = buf.find(b"\xff\xd8")
        if i < 0:
            buf = buf[-3:]
            break
        j = buf.find(b"\xff\xd9", i + 2)
        if j < 0:
            buf = buf[i:]
            break
        buf = buf[j + 2:]
        frames += 1
        last_frame = time.time()
    t = time.time() - t0
    if t >= snap:
        age = (time.time() - last_frame) if last_frame else -1
        print("t=%5.1f  bytes=%8d  reads=%3d  frames=%3d  latest_frame_age=%4.1fs"
              % (t, total, nreads, frames, age))
        snap += 2

sales = sum(read_sizes) / max(1, nreads)
print("END: %.1fs total=%d reads=%d avg_read=%d frames=%d" %
      (time.time() - t0, total, nreads, int(sales), frames))
