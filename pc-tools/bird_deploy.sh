#!/bin/sh
# birdboard pipeline deploy + verify (PYTHON birdboard era).
# The /det.jpg + /frame.jpg byte checks below are python-era endpoints;
# the C birdcpp answers them 410 Gone on purpose (live-only, no snapshot
# writes) - it is verified via /stream + /phone instead, see cpp/README.md.
echo "== copy pumper into bird =="
/bin/cp /home/root/camkit/frame_pumper /home/root/bird/frame_pumper
ls -l /home/root/bird/frame_pumper | cut -d' ' -f5
echo "== kill old capture loops (numbered snapshot writer) =="
/usr/bin/pkill -f live_detect.sh 2>/dev/null
echo "crontab:"
crontab -l 2>/dev/null | grep -v "^#" | head -5
echo "== hand over camera to birdboard =="
/bin/systemctl stop camkit
sleep 2
/bin/systemctl is-active camkit birdboard
echo "== wait for CSI session to settle =="
sleep 12
echo "== /phone (backend + fps) =="
curl -s http://127.0.0.1:8888/phone
echo
echo "== /meta =="
curl -s http://127.0.0.1:8888/meta.json
echo
echo "== /det bytes =="
curl -s http://127.0.0.1:8888/det.jpg -o /tmp/d.jpg && ls -l /tmp/d.jpg | cut -d' ' -f5
echo "== /frame bytes =="
curl -s http://127.0.0.1:8888/frame.jpg -o /tmp/f.jpg && ls -l /tmp/f.jpg | cut -d' ' -f5
echo "== disk-write check: file count in /home/root/bird now, +15 s =="
ls /home/root/bird | wc -l
sleep 15
ls /home/root/bird | wc -l
echo "== who holds video0 =="
for p in /proc/[0-9]*; do
  for f in "$p"/fd/*; do
    tgt=$(readlink "$f" 2>/dev/null)
    case "$tgt" in
      */dev/video0*) echo "$(basename $p) $(ps -p $(basename $p) -o comm= 2>/dev/null)" ;;
    esac
  done
done 2>/dev/null
