"""
BirdCNN Model Architecture
==========================
This file defines a custom Convolutional Neural Network (CNN) for bird species classification.

WHAT IS A CNN?
- A CNN is a type of neural network designed specifically for image processing
- It uses "convolutional" layers that slide filters over the image to detect features
- Early layers detect simple features (edges, colors)
- Deeper layers detect complex features (bird beaks, feather patterns)

HOW THIS MODEL WORKS:
1. Input: A 224x224 RGB image (3 color channels: Red, Green, Blue)
2. Feature Extractor: 4 blocks of Conv layers that learn visual patterns
3. Classifier: Fully connected layers that map features to bird species

ARCHITECTURE SUMMARY:
- Block 1: 3 -> 32 channels (detects edges, basic textures)
- Block 2: 32 -> 64 channels (detects patterns, shapes)
- Block 3: 64 -> 128 channels (detects complex patterns)
- Block 4: 128 -> 256 channels (detects high-level features)
- Classifier: 256 -> 512 -> 256 -> num_classes
"""

# PyTorch imports
import torch          # Main PyTorch library for tensor operations and GPU support
import torch.nn as nn # Neural network modules (layers, loss functions, etc.)


class BirdCNN(nn.Module):
    """
    Custom CNN for bird species classification.

    This class inherits from nn.Module, which is the base class for all
    PyTorch neural networks. It provides:
    - Automatic gradient computation (backpropagation)
    - Parameter management
    - GPU support via .to(device)

    INPUT FORMAT:
    - Batch of images: (batch_size, 3, 224, 224)
    - batch_size: Number of images processed at once
    - 3: RGB color channels
    - 224x224: Image height x width (pixels)

    OUTPUT FORMAT:
    - Class logits: (batch_size, num_classes)
    - Each value represents the model's "confidence" for that class
    - Higher value = more confident prediction
    """

    def __init__(self, num_classes=200):
        """
        Initialize the BirdCNN model.

        Args:
            num_classes (int): Number of bird species to classify.
                              Default is 200 (common for CUB-200 dataset).
                              Change this to match YOUR number of bird species.
        """
        # Call the parent class (nn.Module) constructor
        # This is REQUIRED for all PyTorch models
        super(BirdCNN, self).__init__()

        # ============================================================
        # FEATURE EXTRACTOR
        # This part of the network learns to extract features from images
        # It uses convolutional layers to detect patterns
        # ============================================================
        self.features = nn.Sequential(
            # --------------------------------------------------------
            # BLOCK 1: Input (3 channels) -> Output (32 channels)
            # Input:  224 x 224 x 3  (RGB image)
            # Output: 112 x 112 x 32 (32 feature maps)
            # Purpose: Detect basic features like edges and colors
            # --------------------------------------------------------

            # Conv2d: Applies 32 filters (kernels) of size 3x3 to the input
            # - in_channels=3: Input has 3 channels (RGB)
            # - out_channels=32: We want 32 output feature maps
            # - kernel_size=3: Each filter is 3x3 pixels
            # - padding=1: Adds 1 pixel border to keep output same size
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),

            # BatchNorm2d: Normalizes the output of convolution
            # - Why? It stabilizes training by keeping values in a consistent range
            # - It also allows higher learning rates = faster training
            nn.BatchNorm2d(32),

            # ReLU: Applies rectified linear unit activation
            # - Formula: f(x) = max(0, x)
            # - Why? Adds non-linearity so model can learn complex patterns
            # - inplace=True: Modifies input directly (saves memory)
            nn.ReLU(inplace=True),

            # Second convolution in Block 1 (more feature extraction)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # MaxPool2d: Reduces spatial dimensions by taking maximum value
            # - kernel_size=2: Takes maximum over 2x2 windows
            # - Output size: 112 -> 56 (halves width and height)
            # - Why? Reduces computation, adds translation invariance
            nn.MaxPool2d(2),

            # Dropout2d: Randomly zeros entire feature maps during training
            # - p=0.25: 25% of feature maps are set to zero
            # - Why? Prevents overfitting (memorizing training data)
            # - Only active during training, disabled during inference
            nn.Dropout2d(0.25),

            # --------------------------------------------------------
            # BLOCK 2: 32 channels -> 64 channels
            # Input:  112 x 112 x 32
            # Output: 56 x 56 x 64
            # Purpose: Detect patterns, textures, simple shapes
            # --------------------------------------------------------
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # --------------------------------------------------------
            # BLOCK 3: 64 channels -> 128 channels
            # Input:  56 x 56 x 64
            # Output: 28 x 28 x 128
            # Purpose: Detect complex patterns, feather arrangements
            # --------------------------------------------------------
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # --------------------------------------------------------
            # BLOCK 4: 128 channels -> 256 channels
            # Input:  28 x 28 x 128
            # Output: 1 x 1 x 256
            # Purpose: Detect high-level features (bird parts)
            # --------------------------------------------------------
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # AdaptiveAvgPool2d: Averages each feature map to a single value
            # - (1, 1): Output will be 1x1 per feature map
            # - Input: 28 x 28 x 256
            # - Output: 1 x 1 x 256 (global average pooling)
            # - Why? Produces fixed-size output regardless of input size
            # - This is the bridge between conv layers and classifier
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ============================================================
        # CLASSIFIER (Fully Connected Layers)
        # This part takes the extracted features and predicts the class
        # ============================================================
        self.classifier = nn.Sequential(
            # Flatten: Converts 3D feature map to 1D vector
            # - Input: (batch, 256, 1, 1)
            # - Output: (batch, 256)
            nn.Flatten(),

            # First fully connected layer
            # - 256 input features -> 512 output features
            # - This layer learns combinations of features
            nn.Dropout(0.5),          # 50% dropout (heavy regularization)
            nn.Linear(256, 512),       # Linear transformation: y = xW^T + b
            nn.ReLU(inplace=True),     # Non-linearity

            # Second fully connected layer
            # - 512 input features -> 256 output features
            nn.Dropout(0.3),          # 30% dropout
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            # Output layer
            # - 256 input features -> num_classes output features
            # - NO activation here (raw logits)
            # - CrossEntropyLoss in training applies softmax internally
            nn.Dropout(0.2),          # 20% dropout
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass: Process input through the network.

        This method is called automatically when you do: output = model(input)

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output tensor of shape (batch_size, num_classes)
            These are "logits" (raw scores), NOT probabilities.
            Use torch.softmax() or CrossEntropyLoss to convert to probabilities.
        """
        # Pass input through feature extractor (conv layers)
        x = self.features(x)

        # Pass features through classifier (fully connected layers)
        x = self.classifier(x)

        return x


def get_model(num_classes=200):
    """
    Factory function to create a BirdCNN model.

    This is a convenience function that creates and returns a new model.
    Using a function makes it easier to change the model architecture later.

    Args:
        num_classes (int): Number of bird species to classify

    Returns:
        BirdCNN model instance (not on GPU yet, call .to(device) to move)
    """
    model = BirdCNN(num_classes=num_classes)
    return model


# ============================================================
# TEST CODE: Run this file directly to verify model works
# ============================================================
if __name__ == "__main__":
    # Create a model with 200 bird species
    model = get_model(num_classes=200)

    # Create a fake batch of 2 images (3 channels, 224x224 pixels)
    # This simulates loading 2 bird images
    x = torch.randn(2, 3, 224, 224)  # Random values (doesn't matter for test)

    # Run the images through the model
    output = model(x)

    # Print shapes to verify everything works
    print(f"Input shape:  {x.shape}")   # Should be: torch.Size([2, 3, 224, 224])
    print(f"Output shape: {output.shape}")  # Should be: torch.Size([2, 200])
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Parameters count shows how many weights the model has to learn
    # More parameters = more capacity to learn, but slower training
