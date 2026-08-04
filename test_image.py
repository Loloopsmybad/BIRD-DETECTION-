"""
Test bird detection on a single image.
Put your image path below and run: python test_image.py
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

# ==================== PUT YOUR IMAGE PATH HERE ====================
IMAGE_PATH = r"D:\samsung\BIRD-DETECTION\data_detection\val\images\24_424.jpg"  # <-- CHANGE THIS
# =================================================================

frame = cv2.imread(IMAGE_PATH)
if frame is None:
    print(f"ERROR: Cannot read {IMAGE_PATH}")
    sys.exit(1)

# Run detection
results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)

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

print(f"Found {bird_count} bird(s)")
cv2.imwrite("result.jpg", frame)
print("Result saved to result.jpg")
