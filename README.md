

https://github.com/user-attachments/assets/917be287-1947-4b51-a0ee-d099962f12d6

# Bird Detection

A deep learning project that **detects** birds in images and videos (bounding boxes + confidence) using **YOLOv8-nano**, fine-tuned on Google Colab (T4 GPU) and run locally.

## Demo

<video src="bird.mp4" controls width="640"></video>

> **Note**: GitHub only renders videos under 100MB. If this shows "video too large", compress `bird.mp4` (e.g. `ffmpeg -i bird.mp4 -vcodec libx264 -crf 28 -preset fast bird_small.mp4`) and rename it to `bird.mp4`.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Folder Structure](#project-folder-structure)
3. [Dataset Structure](#dataset-structure)
4. [Installation](#installation)
5. [Training on Colab](#training-on-colab)
6. [Inference](#inference)
7. [Files Description](#files-description)
8. [Models Used](#models-used)
9. [Tips for Better Accuracy](#tips)
10. [Quick Start Commands](#quick-start-commands)

---

## Project Overview

This project uses **Supervised Learning** with **YOLOv8-nano** (a lightweight, fast object detector) to **detect** birds.

### Classification vs Detection
| Task | Question | Output |
|------|----------|--------|
| **Classification** | "What is in this image?" | Class label (e.g., "Bird") |
| **Detection** (this project) | "Where is the bird?" | Bounding boxes + confidence |

The model draws a bounding box around every detected bird with a confidence score.

### Why YOLOv8-nano?
| Approach | Problem |
|----------|---------|
| Custom CNN | Detected birds but boxes were imprecise |
| Teachable Machine (Keras) | Fast but classification only (no boxes) |
| **YOLOv8-nano** (this project) | Fast **AND** accurate bounding boxes (best of both) |

### Key Concepts
| Term | Meaning |
|------|---------|
| **Bounding Box** | Rectangle around the detected bird (x1, y1, x2, y2) |
| **Confidence** | Probability (0-1) that a box contains a bird |
| **IoU** | Intersection over Union - measures box overlap accuracy |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | Number of images processed at once before updating weights |
| **mAP** | Mean Average Precision - standard detection accuracy metric |
| **imgsz** | Inference resolution (640 = accurate, 320 = fast) |

---

## Project Folder Structure

```
D:\samsung\BIRD-DETECTION\
│
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
│
├── Bird_Detection_Colab.ipynb      # YOLOv8 training notebook (Colab)
│
├── camera_inference.py             # Webcam detection (real-time)
├── test_image.py                   # Single-image detection
├── test_video.py                   # Video detection (real-time preview)
│
├── data_detection/                 # Dataset (YOLO format)
│   ├── train/
│   │   ├── images/                 # 760 training images
│   │   └── labels/                 # Labels (class 0 = bird)
│   └── val/
│       ├── images/                 # 190 validation images
│       └── labels/
│
├── checkpoints/
│   └── best.pt                     # Trained YOLOv8-nano model (~6MB)
│
└── venv/                           # Python virtual environment
```

---

## Dataset Structure

The dataset uses **YOLO format**:

```
data_detection/
├── train/
│   ├── images/          # Training images (.jpg)
│   └── labels/          # Labels (.txt, same name as image)
├── val/
│   ├── images/
│   └── labels/
```

### Label File Format
Each `.txt` line (one per bird):
```
class_id x_center y_center width height
```
Example:
```
0 0.567 0.234 0.110 0.090
```
- `class_id` = 0 (bird)
- Coordinates are **normalized** (0-1), relative to image size

> **Note**: The original dataset used class ID `24`. Since we only have one class, all labels are fixed to `0`. The notebook does this automatically.

### Source Dataset
- **Dataset**: [samuelayman/bird-dataset](https://www.kaggle.com/datasets/samuelayman/bird-dataset) on Kaggle
- **Size**: 950 images, 1 class ("bird")
- **Split**: 760 train / 190 val (80/20)

---

## Installation

### Local Machine
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python
```

### Requirements
| Package | Purpose |
|---------|---------|
| ultralytics | YOLOv8 model (train + inference) |
| opencv-python | Camera, video, image drawing |

---

## Training on Colab

The `Bird_Detection_Colab.ipynb` notebook handles everything:

### Setup (one-time)
1. Upload `data_detection/` to Google Drive > `BIRD` folder
2. Runtime → Change runtime type → **GPU (T4)**
3. Run all cells in order

### Notebook Cells (in order)
1. **Setup** - Mount Drive, copy data to local Colab disk, fix labels (24→0)
2. **Install** - `pip install ultralytics`
3. **Create YAML** - Generate `bird_data.yaml` (1 class: bird)
4. **Train** - Fine-tune YOLOv8-nano (50 epochs, imgsz 640, batch 16)
5. **Save Model** - Copy `best.pt` to Drive (persists after runtime ends)

### Inference cells
6. **Single Image** - Detect birds in one image
7. **Batch** - Run detection on the whole val folder

### Why copy to local disk?
- Reading from Google Drive is slow (~10+ min for a copy that takes 30 sec locally)
- The notebook copies data to Colab's local SSD first → fast training, no "corrupt image" errors

### Training Speed
- T4 GPU: ~35-40 sec/epoch, 50 epochs ≈ 30 min
- YOLOv8-nano is small (~3.2M params) so it trains fast

---

## Inference

All scripts auto-detect the model at `checkpoints/best.pt`.

```bash
# Webcam (real-time)
python camera_inference.py

# Single image (edit IMAGE_PATH in test_image.py first)
python test_image.py

# Video (real-time preview, edit VIDEO_PATH in test_video.py first)
python test_video.py
```

### Controls
- Press **`q`** to quit camera / video

### Real-time speed tips (CPU)
| Setting | Speed | Accuracy |
|---------|-------|----------|
| `imgsz=640` | baseline | best for small birds |
| `imgsz=320` | ~2x faster | good for large birds |
| GPU (Colab) | ~50+ FPS | best |

### Auto GPU detection
Scripts automatically use CUDA if available, else fall back to CPU:
```python
device = 0 if torch.cuda.is_available() else 'cpu'
```

---

## Files Description

| File | Purpose |
|------|---------|
| `Bird_Detection_Colab.ipynb` | Full YOLOv8 training + inference notebook |
| `camera_inference.py` | Real-time webcam detection |
| `test_image.py` | Single image detection (saves `result.jpg`) |
| `test_video.py` | Video detection with live preview (saves `video_result.mp4`) |
| `checkpoints/best.pt` | Trained YOLOv8-nano model |

---

## Models Used

### YOLOv8-nano (Ultralytics)
- Pretrained on COCO (already knows "bird"), fine-tuned on our dataset
- **Task**: Detection (bounding boxes)
- **Parameters**: ~3.2M
- **File size**: ~6MB
- **Training**: 50 epochs on Colab T4 (~30 min)
- **Speed**: Real-time on CPU, very fast on GPU

---

## Tips for Better Accuracy

1. **More data**: 950 images is small - collect more bird images
2. **Train longer**: Try 100+ epochs with early stopping
3. **Bigger model**: Try YOLOv8s / YOLOv8m for higher accuracy (slower)
4. **Higher resolution**: `imgsz=640` captures small birds better
5. **Data augmentation**: ultralytics has built-in augmentation (flips, HSV, mosaic)
6. **Bad detections**: Lower `conf` threshold (e.g. 0.25 → 0.15) to catch more birds, raise it to reduce false positives

---

## Quick Start Commands

```bash
# Webcam detection (real-time)
D:\samsung\BIRD-DETECTION\venv\Scripts\python.exe camera_inference.py

# Single image test
D:\samsung\BIRD-DETECTION\venv\Scripts\python.exe test_image.py

# Video detection
D:\samsung\BIRD-DETECTION\venv\Scripts\python.exe test_video.py
```

---

## License

This project is for educational purposes. Use any bird dataset that permits research use.
