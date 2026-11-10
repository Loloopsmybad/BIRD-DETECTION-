# Training & PC-side model tools

Part of the **Bird Detection** repo. This folder holds everything about the
detection *model itself*: dataset, YOLOv8-nano fine-tuning notebook, the
trained checkpoints, and the small PC-side python scripts used to try the
model against a webcam/image/video before it ever went onto the board.

The video demo referenced by the repo root README lives in `..\media\bird.mp4`.

<video src="../media/bird.mp4" controls width="640"></video>

> **Note**: GitHub only renders videos under 100 MB. If this shows
> "video too large", compress `..\media\bird.mp4` (e.g.
> `ffmpeg -i bird.mp4 -vcodec libx264 -crf 28 -preset fast bird_small.mp4`).

---

## Table of Contents
1. [Model overview](#model-overview)
2. [Dataset structure](#dataset-structure)
3. [Installation](#installation)
4. [Training on Colab](#training-on-colab)
5. [PC-side inference tools](#pc-side-inference-tools)
6. [Checkpoints](#checkpoints)
7. [Tips for better accuracy](#tips)

---

## Model overview

Supervised detection with **YOLOv8-nano**, fine-tuned on Google Colab (T4 GPU).

### Why YOLOv8-nano?
| Approach | Problem |
|----------|---------|
| Custom CNN | Detected birds but boxes were imprecise |
| Teachable Machine (Keras) | Fast but classification only (no boxes) |
| **YOLOv8-nano** (this project) | Fast **AND** accurate bounding boxes (best of both) |

### Key concepts
| Term | Meaning |
|------|---------|
| **Bounding Box** | Rectangle around the detected bird (x1, y1, x2, y2) |
| **Confidence** | Probability (0-1) that a box contains a bird |
| **IoU** | Intersection over Union - measures box overlap accuracy |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | Number of images processed at once before updating weights |
| **mAP** | Mean Average Precision - standard detection accuracy metric |
| **imgsz** | Inference resolution (640 = accurate, 320 = fast) |

The model has exactly **one class**: `bird` (class id 0).

---

## Dataset structure

`data_detection\` uses YOLO format:

```
training\data_detection/
├── train/
│   ├── images/          # 760 training images (.jpg)
│   └── labels/          # Labels (.txt, same base name as image)
└── val/
    ├── images/          # 190 validation images
    └── labels/
```

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

> The original Kaggle dataset used class ID `24`; all labels are fixed to
> `0`. `Bird_Detection_Colab.ipynb` does this automatically on upload.

### Source dataset
- **Dataset**: [samuelayman/bird-dataset](https://www.kaggle.com/datasets/samuelayman/bird-dataset)
- **Size**: 950 images, 1 class ("bird")
- **Split**: 760 train / 190 val (80/20)

---

## Installation

The repo-root `venv\` is the shared python environment for every python
piece of this repo (training tools, camera lab, birdboard, bridges).

| Package | Purpose |
|---------|---------|
| ultralytics | YOLOv8 model (train + inference) |
| opencv-python | Camera, video, image drawing |

---

## Training on Colab

`Bird_Detection_Colab.ipynb` handles everything:

### Setup (one-time)
1. Upload `training\data_detection\` to Google Drive > `BIRD` folder
2. Runtime -> Change runtime type -> **GPU (T4)**
3. Run all cells in order

### Notebook cells (in order)
1. **Setup** - Mount Drive, copy data to local Colab disk, fix labels (24->0)
2. **Install** - `pip install ultralytics`
3. **Create YAML** - Generate `bird_data.yaml` (1 class: bird)
4. **Train** - Fine-tune YOLOv8-nano (50 epochs, imgsz 640, batch 16)
5. **Save Model** - Copy `best.pt` to Drive (persists after runtime ends)

### Inference cells
6. **Single Image** - Detect birds in one image
7. **Batch** - Run detection on the whole val folder

### Why copy to local disk?
Reading from Google Drive is slow (~10+ min for a copy that takes 30 s
locally). The notebook copies data to Colab's local SSD first -> fast
training, no "corrupt image" errors.

### Training speed
- T4 GPU: ~35-40 s/epoch, 50 epochs ≈ 30 min
- YOLOv8-nano is small (~3.2M params) so it trains fast

---

## PC-side inference tools

`tools\` - quick sanity checks of the model on the PC (before/alongside
the on-board live systems):

```
# Webcam (real-time)
..\venv\Scripts\python.exe tools\camera_inference.py

# Single image (edit IMAGE_PATH in the script first)
..\venv\Scripts\python.exe tools\test_image.py

# Video (real-time preview; saves ..\media\video_result.mp4)
..\venv\Scripts\python.exe tools\test_video.py
```

- Press **`q`** to quit camera / video windows.
- Scripts auto-detect the model at `checkpoints\best.pt` and use CUDA
  when available, else CPU:

```python
device = 0 if torch.cuda.is_available() else 'cpu'
```

| Setting | Speed | Accuracy |
|---------|-------|----------|
| `imgsz=640` | baseline | best for small birds |
| `imgsz=320` | ~2x faster | good for large birds |
| GPU | ~50+ FPS | best |

---

## Checkpoints

| file | role |
|---|---|
| `checkpoints\best.pt` | trained YOLOv8-nano weights (~6 MB, ultralytics form) |
| `checkpoints\best.onnx` | exported ONNX - **this is what the board runs** (same file that ships inside `..\python\` and on the board at `/home/root/birdcpp/best.onnx`) |

---

## Tips for better accuracy {#tips}

1. **More data**: 950 images is small - collect more bird images
2. **Train longer**: Try 100+ epochs with early stopping
3. **Bigger model**: Try YOLOv8s / YOLOv8m for higher accuracy (slower)
4. **Higher resolution**: `imgsz=640` captures small birds better
5. **Data augmentation**: ultralytics has built-in augmentation (flips, HSV, mosaic)
6. **Bad detections**: Lower `conf` threshold (e.g. 0.25 -> 0.15) to catch more birds, raise it to reduce false positives
