"""
Bird detection on a video - REAL-TIME preview.
Plays the video with detection boxes drawn live.
Put your video path below and run: python test_video.py
"""

import os
import sys
import time
import cv2

# Always resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints', 'best.pt')

# ==================== CHECK DEPENDENCIES ====================
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system(f'"{sys.executable}" -m pip install ultralytics')
    from ultralytics import YOLO

# ==================== LOAD TRAINED MODEL ====================
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Make sure checkpoints/best.pt exists")
    sys.exit(1)

# Use GPU if available, else CPU
import torch
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# ==================== PUT YOUR VIDEO PATH HERE ====================
VIDEO_PATH = r"D:\samsung\BIRD-DETECTION\video.mp4"  # <-- CHANGE THIS
# =================================================================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video fps: {fps:.1f}. Press 'q' to quit.")

# Store annotations as we go (save at end, doesn't slow playback)
all_frames = []
frame_count = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fast inference (smaller imgsz = faster)
    results = model.predict(frame, conf=0.25, imgsz=416, device=device, verbose=False)

    # Draw detections
    bird_count = 0
    for r in results:
        boxes = r.boxes
        bird_count = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Bird: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show processing speed
    now = time.time()
    proc_fps = 1.0 / max(now - prev_time, 0.001)
    prev_time = now
    cv2.putText(frame, f"Birds: {bird_count} | FPS: {proc_fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    all_frames.append(frame.copy())
    frame_count += 1

    cv2.imshow('Bird Detection (Real-time)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nProcessed {frame_count} frames")

# Save annotated video after playback
OUTPUT_PATH = "video_result.mp4"
if all_frames:
    h, w = all_frames[0].shape[:2]
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in all_frames:
        writer.write(f)
    writer.release()
    print(f"Annotated video saved to {OUTPUT_PATH}")
