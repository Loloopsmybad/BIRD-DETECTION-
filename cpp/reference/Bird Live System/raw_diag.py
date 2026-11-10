import socket, sys, time

ua = sys.argv[1] if len(sys.argv) > 1 else "board-bird/1.0"
s = socket.create_connection(("192.168.1.3", 4747), 8)
s.sendall(("GET /video HTTP/1.1\r\nHost: 192.168.1.3:4747\r\n"
           "User-Agent: %s\r\nAccept: */*\r\n\r\n" % ua).encode())
s.settimeout(10)
buf = b""
t0 = time.time()
closed = False
while time.time() - t0 < 15:
    try:
        d = s.recv(65536)
    except socket.timeout:
        print("RECV TIMEOUT at t=%.1f" % (time.time() - t0))
        break
    if not d:
        closed = True
        print("CONN CLOSED BY PHONE at t=%.1f" % (time.time() - t0))
        break
    buf += d
s.close()
soi = buf.count(b"\xff\xd8")
eoi = buf.count(b"\xff\xd9")
i = buf.find(b"\xff\xd8")
j = buf.find(b"\xff\xd9", i + 2) if i >= 0 else -1
first = (j - i + 2) if (i >= 0 and j > i) else -1
after_eoi = len(buf) - (j + 2) if (i >= 0 and j > i) else 0
print("UA=%s total_bytes=%d closed=%s" % (ua, len(buf), closed))
print("SOI=%d EOI=%d first_jpeg=%d bytes_so_far_after_first_eoi=%d" % (soi, eoi, first, after_eoi))
head = buf[:200]
print("HEAD:", head[:160])
if soi >= 2:
    i2 = buf.find(b"\xff\xd8", i + 2)
    print("2nd frame starts at", i2, "- gap since 1st eoi:", i2 - (j + 2))
