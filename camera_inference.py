"""
Bird Detection - Camera Inference (YOLOv8)
===========================================
Uses OpenCV webcam to detect birds in real-time using trained YOLOv8 model.

HOW IT WORKS:
1. Open webcam using OpenCV VideoCapture
2. Each frame is passed through YOLOv8 model
3. Model returns bounding boxes + confidence for each detected bird
4. Draw green rectangles around detected birds on the frame
5. Press 'q' to quit

USAGE:
    python camera_inference.py
"""

import os
import sys
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

model = YOLO(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# ==================== OPEN CAMERA ====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera 0 failed, trying camera 1...")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No camera found.")
    sys.exit(1)
print("Camera opened. Press 'q' to quit.")

# ==================== MAIN LOOP ====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 detection
    results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)

    # Draw detections on frame
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

    cv2.putText(frame, f"Birds: {bird_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Bird Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
