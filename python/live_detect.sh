#!/bin/sh
# live_detect.sh — board-side loop: pull phone frame from PC bridge, run YOLO, republish
# runs on the board; expects /home/root/bird/{board_detection.py,best.onnx,frame.jpg}
cd /home/root/bird

# keep the results web server alive (serves annotated results to the PC)
( python3 -m http.server 8888 >/dev/null 2>&1 & ) 2>/dev/null

seq=0
while true; do
  if wget -q -T 6 "http://192.168.1.5:9000/frame.jpg" -O frame.jpg 2>/dev/null; then
    python3 board_detection.py frame.jpg --save > meta.json 2>/dev/null
    if [ -f frame_det.jpg ]; then
      mv frame_det.jpg live_det.jpg 2>/dev/null
      cp live_det.jpg "live_${seq}_det.jpg" 2>/dev/null
      old=$((seq-30)); [ $old -ge 0 ] && rm -f "live_${old}_det.jpg"
      seq=$((seq+1))
    fi
  fi
  sleep 1
done
