"""
Bird Dataset Loading and Preprocessing
=======================================
This file handles loading bird images from disk and preparing them for training.

WHAT THIS FILE DOES:
1. Reads images from folders (one folder per bird species)
2. Applies transformations (resize, augmentation, normalization)
3. Creates DataLoader objects that feed batches of images to the model

DATA AUGMENTATION EXPLAINED:
- During training, we apply random transformations to images
- This artificially increases dataset size and variety
- Makes the model more robust to different lighting, angles, etc.

NORMALIZATION EXPLAINED:
- ImageNet pre-trained models expect images normalized with:
  - mean = [0.485, 0.456, 0.406]  (RGB channels)
  - std  = [0.229, 0.224, 0.225]  (RGB channels)
- This converts pixel values from [0, 1] range to approximately [-1, 1]
"""

import os                          # For file path operations
from PIL import Image              # Python Imaging Library - for loading images
from torch.utils.data import Dataset, DataLoader  # PyTorch data loading utilities
from torchvision import transforms # Image transformation utilities


class BirdDataset(Dataset):
    """
    Custom PyTorch Dataset for bird images.

    A Dataset class must implement:
    - __len__(): Returns the total number of samples
    - __getitem__(idx): Returns a single sample (image, label)

    HOW PYTORCH DATASETS WORK:
    1. Dataset loads one image at a time when requested
    2. DataLoader groups images into batches for efficient training
    3. This is memory-efficient (doesn't load all images at once)

    EXPECTED FOLDER STRUCTURE:
    data/
    ├── robin/
    │   ├── img001.jpg
    │   ├── img002.jpg
    │   └── ...
    ├── cardinal/
    │   ├── img001.jpg
    │   └── ...
    └── ...
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset by scanning the folder structure.

        Args:
            root_dir (str): Path to the root folder containing class subfolders.
                           Example: "data/train" which contains "robin/", "cardinal/", etc.
            transform: Optional torchvision transforms to apply to images.
                      If None, images are loaded but NOT transformed.
        """
        self.root_dir = root_dir      # Store the root directory path
        self.transform = transform    # Store the transform pipeline

        # --------------------------------------------------------
        # STEP 1: Find all class names (subfolder names)
        # --------------------------------------------------------
        # os.listdir() returns all items in a directory
        # os.path.isdir() checks if an item is a folder (not a file)
        # sorted() ensures consistent class ordering
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # --------------------------------------------------------
        # STEP 2: Create a mapping from class name to numeric label
        # --------------------------------------------------------
        # Example: {"cardinal": 0, "robin": 1, "sparrow": 2}
        # Neural networks work with numbers, not strings
        # So we convert species names to integers
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # --------------------------------------------------------
        # STEP 3: Build a list of all image paths and their labels
        # --------------------------------------------------------
        # Each entry: (full_path_to_image, numeric_label)
        # Example: ("data/train/robin/img001.jpg", 1)
        self.samples = []

        for cls in self.classes:
            # Get the full path to this class's folder
            cls_dir = os.path.join(root_dir, cls)

            # Loop through all files in this class folder
            for img_name in os.listdir(cls_dir):
                # Only keep image files (skip hidden files, txt files, etc.)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    # Store the full path and the numeric label
                    self.samples.append((
                        os.path.join(cls_dir, img_name),  # Full image path
                        self.class_to_idx[cls]              # Numeric label
                    ))

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        This is called by PyTorch when you do: len(dataset)
        Required for DataLoader to know when to stop iterating.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample (image + label) from the dataset.

        This is called by PyTorch when DataLoader requests a specific sample.

        Args:
            idx (int): Index of the sample to retrieve (0 to len-1)

        Returns:
            tuple: (image_tensor, label_int)
                   - image_tensor: Image as PyTorch tensor with shape (3, 224, 224)
                   - label_int: Integer label (e.g., 0 for cardinal, 1 for robin)
        """
        # Get the image path and label for this index
        img_path, label = self.samples[idx]

        # Load the image from disk
        # .convert('RGB') ensures the image has 3 channels (not RGBA or grayscale)
        image = Image.open(img_path).convert('RGB')

        # Apply the transform pipeline if one was provided
        # Transforms convert PIL Image -> PyTorch Tensor and apply augmentations
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=224, is_training=True):
    """
    Create image transformation pipelines.

    TRANSFORMS EXPLAINED:
    - Transforms are applied sequentially to each image
    - They convert raw images to the format the model expects
    - During training, we add random augmentations to increase variety

    Args:
        image_size (int): Target size for images (images will be resized to this)
        is_training (bool): If True, apply data augmentation (random flips, etc.)
                           If False, only resize and normalize (for validation/testing)

    Returns:
        transforms.Compose: A pipeline of transforms to apply to each image
    """

    if is_training:
        # --------------------------------------------------------
        # TRAINING TRANSFORMS (with data augmentation)
        # These add random variations to prevent overfitting
        # --------------------------------------------------------
        return transforms.Compose([
            # Resize: Make image slightly larger than target size
            # Why 256x256? So we can randomly crop to 224x224
            transforms.Resize((image_size + 32, image_size + 32)),

            # RandomCrop: Take a random 224x224 patch from the 256x256 image
            # This teaches the model to recognize birds regardless of position
            transforms.RandomCrop(image_size),

            # RandomHorizontalFlip: 50% chance of flipping image left-right
            # Birds can face either direction, so this is realistic
            transforms.RandomHorizontalFlip(p=0.5),

            # RandomVerticalFlip: 10% chance of flipping image upside-down
            # Less common but sometimes birds are photographed at odd angles
            transforms.RandomVerticalFlip(p=0.1),

            # RandomRotation: Rotate image by random angle between -30 and +30 degrees
            # Birds aren't always perfectly upright in photos
            transforms.RandomRotation(30),

            # ColorJitter: Randomly change brightness, contrast, saturation
            # Simulates different lighting conditions
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),

            # ToTensor: Convert PIL Image to PyTorch Tensor
            # - Changes format from (H, W, C) to (C, H, W)
            # - Scales pixel values from [0, 255] to [0.0, 1.0]
            transforms.ToTensor(),

            # Normalize: Standardize image using ImageNet statistics
            # - mean=[0.485, 0.456, 0.406]: Average color of ImageNet images
            # - std=[0.229, 0.224, 0.225]: Standard deviation of ImageNet images
            # - This helps the model learn faster and more reliably
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # --------------------------------------------------------
        # VALIDATION/TEST TRANSFORMS (no augmentation)
        # We don't want random changes during evaluation
        # --------------------------------------------------------
        return transforms.Compose([
            # Resize: Simply resize to target size (no random crop)
            transforms.Resize((image_size, image_size)),

            # ToTensor: Convert to tensor
            transforms.ToTensor(),

            # Normalize: Same normalization as training
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloaders(data_dir, batch_size=32, image_size=224, num_workers=2):
    """
    Create DataLoaders for training, validation, and testing.

    DATALOADER EXPLAINED:
    - DataLoader wraps a Dataset and provides:
      - Batching: Groups multiple images together
      - Shuffling: Randomizes order each epoch (training only)
      - Parallel loading: Uses multiple CPU threads to load images
    - This is essential for efficient GPU training

    Args:
        data_dir (str): Root data directory containing train/val/test folders
        batch_size (int): Number of images per batch
                         - Larger = faster training but more GPU memory
                         - Smaller = slower but uses less memory
        image_size (int): Target image size (224x224)
        num_workers (int): Number of parallel data loading threads
                          - 0 = load in main process (slow)
                          - 2-4 = good for most systems
                          - More = faster but uses more CPU

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
            - train_loader: DataLoader for training (shuffled)
            - val_loader: DataLoader for validation (not shuffled)
            - test_loader: DataLoader for testing (not shuffled)
            - class_names: List of species names (e.g., ["cardinal", "robin", ...])
    """

    # Create dataset objects for each split
    # Each dataset scans its folder and builds a list of (image_path, label) pairs
    train_dataset = BirdDataset(
        os.path.join(data_dir, 'train'),       # Path: data/train/
        transform=get_transforms(image_size, is_training=True)  # With augmentation
    )

    val_dataset = BirdDataset(
        os.path.join(data_dir, 'val'),          # Path: data/val/
        transform=get_transforms(image_size, is_training=False)  # No augmentation
    )

    test_dataset = BirdDataset(
        os.path.join(data_dir, 'test'),         # Path: data/test/
        transform=get_transforms(image_size, is_training=False)  # No augmentation
    )

    # Create DataLoader objects
    # DataLoader iterates over the dataset and yields batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,   # How many images per batch
        shuffle=True,            # Shuffle training data each epoch (important!)
        num_workers=num_workers  # Parallel loading threads
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,           # Don't shuffle validation (consistent evaluation)
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,           # Don't shuffle test data
        num_workers=num_workers
    )

    # Return the class names (species names in alphabetical order)
    return train_loader, val_loader, test_loader, train_dataset.classes


# ============================================================
# TEST CODE: Run this file directly to verify data loading works
# ============================================================
if __name__ == "__main__":
    # Try loading data from the "data" folder
    # This will fail if you haven't set up your data yet
    print("Testing dataset loading...")
    print("Note: This will fail if data/ folder is empty.\n")

    try:
        train_loader, val_loader, test_loader, classes = get_dataloaders(
            "data",     # Root data directory
            batch_size=4  # Small batch for testing
        )

        print(f"Number of bird species found: {len(classes)}")
        print(f"Class names: {classes}")

        # Get one batch from the training loader
        images, labels = next(iter(train_loader))
        print(f"\nBatch image shape: {images.shape}")
        # Should be: torch.Size([4, 3, 224, 224])
        # 4 images, 3 channels, 224x224 pixels

        print(f"Batch labels shape: {labels.shape}")
        # Should be: torch.Size([4])
        # 4 integer labels

        print(f"Sample labels: {labels}")

    except FileNotFoundError:
        print("ERROR: Could not find 'data' folder or it's empty.")
        print("Please organize your bird images first!")
        print("See README.md for the required folder structure.")
