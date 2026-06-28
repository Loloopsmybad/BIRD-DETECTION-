# Bird Detection using Custom CNN

A deep learning project that classifies bird species from images using a custom Convolutional Neural Network (CNN) built with PyTorch.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Folder Structure](#project-folder-structure)
3. [Architecture](#architecture)
4. [Dataset Structure](#dataset-structure)
5. [Installation](#installation)
6. [Training on Kaggle](#training-on-kaggle)
7. [Files Description](#files-description)
8. [Hyperparameters](#hyperparameters)
9. [Tips for Better Accuracy](#tips)

---

## Project Overview

This project uses **Supervised Learning** with a **CNN (Convolutional Neural Network)** to classify bird species from images. The model takes a 224x224 RGB image as input and outputs the predicted bird species class.

### Why CNN for Bird Classification?
- CNNs automatically learn spatial features (edges, textures, shapes) from images
- Convolutional layers detect local patterns like feather textures, beak shapes, and color patterns
- Pooling layers reduce spatial dimensions while keeping important features
- This is a **Classification** task (predicting discrete species), NOT regression

### Key Concepts
| Term | Meaning |
|------|---------|
| **Supervised Learning** | Training with labeled data (image + species name) |
| **Classification** | Predicting which category/species an image belongs to |
| **CNN** | Neural network with convolutional layers for image processing |
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | Number of images processed at once before updating weights |
| **Learning Rate** | How big of a step the optimizer takes during training |

---

## Project Folder Structure

```
D:\samsung\BIRD-DETECTION\
│
├── .gitignore                  # Git ignore rules (excludes checkpoints/, data/, __pycache__/)
│
├── README.md                   # This file - full project documentation
│
├── requirements.txt            # Python dependencies (torch, torchvision, pillow, etc.)
│
├── train.py                    # Main training script (run this to train the model)
│
├── main.ipynb                  # Jupyter notebook for inference (run this to test predictions)
│
├── models/                     # Model architecture folder
│   ├── __init__.py             # Makes 'models' a Python package (empty file)
│   └── model.py                # BirdCNN class definition (the neural network)
│
├── utils/                      # Utility functions folder
│   ├── __init__.py             # Makes 'utils' a Python package (empty file)
│   └── dataset.py              # Data loading, transforms, and DataLoader creation
│
├── data/                       # Dataset folder (YOU must organize your images here)
│   ├── train/                  # Training images (70-80% of your data)
│   │   ├── species_name_1/     # One subfolder per bird species
│   │   │   ├── img_001.jpg     # Individual bird images
│   │   │   ├── img_002.jpg
│   │   │   └── ...
│   │   ├── species_name_2/
│   │   └── ...
│   │
│   ├── val/                    # Validation images (10-15% of your data)
│   │   ├── species_name_1/     # Same species folders as train/
│   │   ├── species_name_2/
│   │   └── ...
│   │
│   └── test/                   # Test images (10-15% of your data)
│       ├── species_name_1/     # Same species folders as train/
│       ├── species_name_2/
│       └── ...
│
└── checkpoints/                # Saved model weights (created during training)
    ├── best_model.pth          # Best model based on validation accuracy
    └── final_model.pth         # Final model after all epochs
```

### What Each Folder Does

| Folder | Purpose | When Created |
|--------|---------|--------------|
| `models/` | Stores the neural network architecture code | You create this |
| `utils/` | Stores helper code (data loading, transforms) | You create this |
| `data/` | Stores all bird images organized by species | You organize this |
| `data/train/` | Images used to train the model (model sees these) | You organize this |
| `data/val/` | Images used to check progress during training | You organize this |
| `data/test/` | Images used for final evaluation (never seen by model) | You organize this |
| `checkpoints/` | Stores saved model weights (.pth files) | Created during training |

### What Each File Does

| File | Lines | Purpose |
|------|-------|---------|
| `train.py` | ~157 | Main entry point - run this to train the model |
| `models/model.py` | ~86 | Defines the BirdCNN architecture (layers, forward pass) |
| `utils/dataset.py` | ~101 | Handles loading images, applying transforms, batching |
| `main.ipynb` | ~92 | Notebook for testing predictions on new images |
| `requirements.txt` | ~6 | Lists all Python packages needed |
| `.gitignore` | ~7 | Tells Git which files/folders to ignore |
| `README.md` | ~255 | This documentation file |

### Data Flow

```
YOUR IMAGES (organized in data/ folder)
        │
        ▼
  dataset.py (loads images, applies transforms)
        │
        ▼
  train.py (feeds batches to model, computes loss, updates weights)
        │
        ▼
  model.py (BirdCNN learns to extract features and classify)
        │
        ▼
  checkpoints/best_model.pth (saved trained weights)
        │
        ▼
  main.ipynb (loads weights, predicts on new images)
```

---

## Architecture

```
BirdCNN Architecture (Input: 224x224x3)
========================================

FEATURE EXTRACTOR (Convolutional Layers):
-----------------------------------------
Block 1: Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool -> Dropout
  - Detects: edges, basic textures, color gradients
  - Output: 112x112x32

Block 2: Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool -> Dropout
  - Detects: patterns, textures, simple shapes
  - Output: 56x56x64

Block 3: Conv(64->128) -> BN -> ReLU -> Conv(128->128) -> BN -> ReLU -> MaxPool -> Dropout
  - Detects: complex patterns, feather arrangements
  - Output: 28x28x128

Block 4: Conv(128->256) -> BN -> ReLU -> Conv(256->256) -> BN -> ReLU -> AdaptiveAvgPool
  - Detects: high-level features, bird parts (beak, wings, tail)
  - Output: 1x1x256

CLASSIFIER (Fully Connected Layers):
-------------------------------------
Flatten -> Dropout(0.5) -> Linear(256->512) -> ReLU -> Dropout(0.3)
       -> Linear(512->256) -> ReLU -> Dropout(0.2) -> Linear(256->num_classes)

Total Parameters: ~2.8M
```

### Layer-by-Layer Explanation

| Layer | Purpose | Why Used |
|-------|---------|----------|
| `Conv2d` | Extracts features from image patches | Learns spatial patterns like edges, textures |
| `BatchNorm2d` | Normalizes activations | Stabilizes training, allows higher learning rates |
| `ReLU` | Adds non-linearity | Without this, model can only learn linear relationships |
| `MaxPool2d` | Reduces spatial dimensions | Reduces computation, adds translation invariance |
| `Dropout2d` | Randomly zeros feature maps | Prevents overfitting during training |
| `AdaptiveAvgPool2d` | Fixed-size output (1x1) | Allows any input size, produces global features |
| `Linear` | Fully connected classification | Maps features to class scores |
| `Dropout` | Randomly zeros neurons | Prevents overfitting in classifier |

---

## Dataset Structure

Your data folder must be organized like this:

```
data/
├── train/                    # Training images (70-80% of data)
│   ├── species_name_1/       # One folder per bird species
│   │   ├── img_001.jpg       # Images of this species
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── species_name_2/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── ...
├── val/                      # Validation images (10-15% of data)
│   ├── species_name_1/
│   ├── species_name_2/
│   └── ...
└── test/                     # Test images (10-15% of data)
    ├── species_name_1/
    ├── species_name_2/
    └── ...
```

### Data Requirements
- **Minimum**: 50-100 images per species for decent results
- **Recommended**: 200-500+ images per species
- **Image formats**: JPG, JPEG, PNG, BMP, WEBP
- **Image size**: Any size (auto-resized to 224x224 during training)

### Recommended Datasets (Kaggle)
| Dataset | Species | Images | Link |
|---------|---------|--------|------|
| CUB-200-2011 | 200 | ~12,000 | Search "CUB-200" on Kaggle |
| Birds-250 | 250 | ~30,000 | Search "birds-250" on Kaggle |
| NABirds | 555 | ~48,000 | Search "NABirds" on Kaggle |
| iNaturalist Birds | 1000+ | 100,000+ | Search "iNaturalist" on Kaggle |

---

## Installation

### Local Machine (for code editing/testing)
```bash
# Clone or download the project
cd D:\samsung\BIRD-DETECTION

# Install dependencies
pip install -r requirements.txt
```

### Kaggle (for training with GPU)
```bash
# In Kaggle notebook, run this in first cell:
!pip install -r requirements.txt
```

---

## Training on Kaggle

### Step-by-Step Guide

**Step 1: Prepare Dataset**
- Download a bird dataset from Kaggle Datasets
- Ensure it follows the folder structure above (train/val/test)

**Step 2: Upload to Kaggle**
- Create a new Kaggle notebook
- Upload `models/model.py`, `utils/dataset.py`, `train.py` to the notebook
- OR add the dataset to your Kaggle notebook as input

**Step 3: Run Training**
```python
# In Kaggle notebook cell:
!python train.py
```

**Step 4: Download Results**
- After training, download `checkpoints/best_model.pth`
- Use `main.ipynb` locally for inference

### Kaggle Settings
- **Enable GPU**: Settings -> Accelerator -> GPU (P100 or T4)
- **Enable Internet**: Settings -> Internet -> On (for pip install)

---

## Files Description

| File | Purpose | What It Does |
|------|---------|--------------|
| `models/model.py` | Model architecture | Defines the BirdCNN neural network |
| `utils/dataset.py` | Data loading | Loads images, applies transforms, creates batches |
| `train.py` | Training script | Main training loop, saves best model |
| `main.ipynb` | Inference demo | Load trained model and predict on new images |
| `requirements.txt` | Dependencies | List of required Python packages |
| `.gitignore` | Git ignore | Files/folders to exclude from version control |
| `checkpoints/` | Saved models | Stores trained model weights (.pth files) |
| `data/` | Dataset | Your bird images organized by species |

---

## Hyperparameters

These are the default settings in `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Number of images per training step |
| `IMAGE_SIZE` | 224 | Input image resolution (224x224) |
| `NUM_EPOCHS` | 50 | Number of complete training passes |
| `LEARNING_RATE` | 0.001 | Step size for optimizer |
| `NUM_WORKERS` | 2 | Parallel data loading threads |

### Tuning Tips
- **Low accuracy?** Try: more epochs, more data, data augmentation
- **Overfitting?** Try: more dropout, less epochs, more data
- **Training too slow?** Try: smaller batch size, fewer workers
- **GPU out of memory?** Try: smaller batch size (16 or 8)

---

## Tips for Better Accuracy

1. **More Data**: At least 100+ images per species
2. **Balanced Classes**: Same number of images per species
3. **Clean Data**: Remove mislabeled or corrupted images
4. **Data Augmentation**: Already built-in (flips, rotation, color jitter)
5. **Transfer Learning**: Replace this CNN with EfficientNet for higher accuracy
6. **Train Longer**: Try 100-200 epochs if you have time
7. **Learning Rate**: Try 0.0005 if training is unstable

---

## Quick Start Commands

```bash
# Test model architecture
python models/model.py

# Test data loading
python utils/dataset.py

# Train the model
python train.py

# Inference (use main.ipynb)
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- Pillow 9.0+
- numpy 1.24+
- matplotlib 3.7+
- tqdm 4.65+

---

## License

This project is for educational purposes. Use any bird dataset that permits research use.
