import http.client, time

c = http.client.HTTPConnection("192.168.1.3", 4747, timeout=8)
c.request("GET", "/video", headers={"User-Agent": "board-bird/1.0"})
r = c.getresponse()
print("STATUS", r.status, r.getheaders()[:4])
try:
    c.sock.settimeout(20)
    print("sock timeout set ok")
except Exception as e:
    print("sock timeout FAILED:", e)

t0 = time.time()
total = 0
chunks = 0
last = 0
while time.time() - t0 < 25:
    try:
        d = r.read(65536)
    except Exception as e:
        print("READ_EXC", repr(e)[:100], "at t=%.1f total=%d" % (time.time() - t0, total))
        break
    if not d:
        print("EOF total=%d" % total)
        break
    total += len(d)
    chunks += 1
    if total - last > 100000:
        last = total
        print("...bytes=%d t=%.1f" % (total, time.time() - t0))
print("DONE chunks=%d bytes=%d in %.1fs" % (chunks, total, time.time() - t0))
