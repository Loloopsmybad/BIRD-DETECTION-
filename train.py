"""
Bird Species Training Script
=============================
This is the main training script that:
1. Loads the bird dataset
2. Creates the CNN model
3. Trains the model on the training set
4. Validates on the validation set after each epoch
5. Saves the best model based on validation accuracy
6. Evaluates final performance on the test set

HOW TRAINING WORKS (High Level):
================================
1. Forward Pass: Feed images through the model -> get predictions
2. Compute Loss: Compare predictions to actual labels -> get error
3. Backward Pass: Compute gradients (how to reduce the error)
4. Update Weights: Adjust model weights using optimizer
5. Repeat until convergence or max epochs reached

TRAINING TERMINOLOGY:
- Epoch: One complete pass through the entire training dataset
- Batch: A subset of images processed together (e.g., 32 images)
- Loss: How wrong the model's predictions are (lower = better)
- Accuracy: Percentage of correct predictions (higher = better)
- Learning Rate: How big of a step the optimizer takes
- Overfitting: Model memorizes training data but fails on new data
- Underfitting: Model is too simple to learn the patterns
"""

import os            # For file path operations
import time          # For timing training
import copy          # For deep copying model weights
import torch         # PyTorch main library
import torch.nn as nn # Neural network modules
import torch.optim as optim  # Optimizers (Adam, SGD, etc.)
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning rate scheduler

# Import our custom modules
from models.model import get_model          # Our BirdCNN model
from utils.dataset import get_dataloaders   # Our data loading utilities


def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    """
    Train the bird classification model.

    This function implements the complete training loop:
    for each epoch:
        1. Train on training set (update weights)
        2. Validate on validation set (check performance)
        3. Save best model
        4. Adjust learning rate if needed

    Args:
        model: The neural network to train (BirdCNN instance)
        train_loader: DataLoader for training images
        val_loader: DataLoader for validation images
        device: Where to run training ("cuda" for GPU, "cpu" for CPU)
        num_epochs (int): How many times to loop through entire dataset
        lr (float): Learning rate (how big steps the optimizer takes)

    Returns:
        tuple: (trained_model, training_history)
            - trained_model: The best version of the model
            - training_history: Dictionary with loss/accuracy per epoch
    """

    # ============================================================
    # STEP 1: SET UP LOSS FUNCTION, OPTIMIZER, AND SCHEDULER
    # ============================================================

    # CROSS ENTROPY LOSS
    # - Standard loss function for classification tasks
    # - Measures how far predictions are from true labels
    # - Lower loss = better predictions
    # - Takes "logits" (raw scores) as input, NOT probabilities
    # - Internally applies softmax to convert logits to probabilities
    criterion = nn.CrossEntropyLoss()

    # ADAM OPTIMIZER
    # - An adaptive learning rate optimizer (generally works well)
    # - Adjusts learning rate for each parameter individually
    # - weight_decay=1e-4: Adds small penalty to large weights (regularization)
    #   This helps prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # REDUCE LR ON PLATEAU SCHEDULER
    # - Automatically reduces learning rate when validation loss stops improving
    # - mode='min': Monitor validation loss (we want it to decrease)
    # - factor=0.5: Multiply learning rate by 0.5 when reducing
    # - patience=3: Wait 3 epochs without improvement before reducing
    # - Why? High LR early (fast learning), low LR later (fine-tuning)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # ============================================================
    # STEP 2: INITIALIZE TRAINING VARIABLES
    # ============================================================

    # Deep copy the model's initial weights
    # We'll restore these if the model gets worse during training
    best_model_wts = copy.deepcopy(model.state_dict())

    # Track the best validation accuracy achieved
    best_acc = 0.0

    # History dictionary to store metrics for each epoch
    # Useful for plotting training curves later
    history = {
        'train_loss': [],   # Training loss per epoch
        'val_loss': [],     # Validation loss per epoch
        'train_acc': [],    # Training accuracy per epoch
        'val_acc': []       # Validation accuracy per epoch
    }

    # ============================================================
    # STEP 3: TRAINING LOOP
    # ============================================================
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # ================================================
        # PHASE 1: TRAINING
        # - Model learns from training data
        # - Weights are updated here
        # ================================================

        # Set model to TRAINING mode
        # - Enables dropout layers (randomly zeros neurons)
        # - Enables batch normalization (uses batch statistics)
        model.train()

        # Initialize counters for this epoch
        running_loss = 0.0    # Accumulated loss (for averaging)
        correct = 0           # Number of correct predictions
        total = 0             # Total number of images processed

        # Loop through all batches in the training set
        for images, labels in train_loader:
            # Move images and labels to GPU (if available)
            # GPU is much faster for matrix operations
            images, labels = images.to(device), labels.to(device)

            # ZERO GRADIENTS
            # - PyTorch accumulates gradients by default
            # - We must zero them before each batch
            # - Otherwise gradients would be sum of all previous batches
            optimizer.zero_grad()

            # FORWARD PASS
            # - Feed images through the model
            # - Get predictions (logits)
            # - Output shape: (batch_size, num_classes)
            outputs = model(images)

            # COMPUTE LOSS
            # - Compare model's predictions to true labels
            # - Loss is a single number representing "how wrong" we are
            loss = criterion(outputs, labels)

            # BACKWARD PASS (BACKPROPAGATION)
            # - Compute gradients of loss with respect to each weight
            # - Tells us how to adjust each weight to reduce loss
            # - This is the "learning" part of training
            loss.backward()

            # UPDATE WEIGHTS
            # - Optimizer adjusts weights based on computed gradients
            # - Adam optimizer uses adaptive learning rates per parameter
            # - This is where the model actually "learns"
            optimizer.step()

            # ACCUMULATE STATISTICS
            # loss.item() converts loss tensor to Python number
            # Multiply by batch size because loss is averaged over batch
            running_loss += loss.item() * images.size(0)

            # Get the class with highest score (the prediction)
            # torch.max returns (values, indices), we need indices
            _, predicted = torch.max(outputs, 1)

            # Count correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate epoch statistics (average loss and accuracy)
        train_loss = running_loss / total
        train_acc = correct / total

        # ================================================
        # PHASE 2: VALIDATION
        # - Check how well model generalizes to unseen data
        # - No weight updates happen here
        # ================================================

        # Set model to EVALUATION mode
        # - Disables dropout (uses all neurons)
        # - Batch normalization uses running statistics
        model.eval()

        # Initialize validation counters
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        # torch.no_grad() disables gradient computation
        # - Saves memory (gradients not needed for validation)
        # - Speeds up computation
        # - Required for proper evaluation
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass only (no backward pass)
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Accumulate validation statistics
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate validation statistics
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        # ================================================
        # PHASE 3: UPDATE AND LOG
        # ================================================

        # Step the learning rate scheduler
        # - If validation loss hasn't improved for 'patience' epochs,
        #   reduce learning rate by factor
        scheduler.step(val_loss)

        # Store metrics in history (for plotting later)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Print epoch results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # ================================================
        # PHASE 4: SAVE BEST MODEL
        # ================================================

        # Check if this is the best model so far
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save checkpoint to disk
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, 'checkpoints/best_model.pth')
            print(f"*** New best model saved! Acc: {best_acc:.4f} ***")

    # ============================================================
    # STEP 4: RESTORE BEST MODEL
    # ============================================================
    # After training, load the best weights (not the last epoch's weights)
    model.load_state_dict(best_model_wts)
    return model, history


def main():
    """
    Main function to run the complete training pipeline.

    This function:
    1. Sets up configuration parameters
    2. Detects available hardware (GPU/CPU)
    3. Loads the dataset
    4. Creates the model
    5. Trains the model
    6. Evaluates on test set
    7. Saves the final model
    """

    # ============================================================
    # CONFIGURATION - Change these settings as needed
    # ============================================================
    DATA_DIR = "data"           # Path to your data folder
    BATCH_SIZE = 32             # Number of images per batch
                                # - Increase if GPU has more memory (64, 128)
                                # - Decrease if GPU runs out of memory (16, 8)
    IMAGE_SIZE = 224            # Input image size (224x224 pixels)
    NUM_EPOCHS = 50             # Number of training passes
                                # - More epochs = longer training
                                # - 50 is a good starting point
    LEARNING_RATE = 0.001       # Step size for optimizer
                                # - Higher = faster but less stable
                                # - Lower = slower but more stable
    NUM_WORKERS = 2             # Number of CPU threads for data loading
                                # - 0 = slow (single thread)
                                # - 2-4 = good for most systems

    # ============================================================
    # DEVICE DETECTION - Use GPU if available, otherwise CPU
    # ============================================================
    # CUDA is NVIDIA's GPU computing platform
    # Training on GPU is 10-100x faster than CPU!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print GPU info if available
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # ============================================================
    # LOAD DATA
    # ============================================================
    print("\nLoading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        DATA_DIR,
        BATCH_SIZE,
        IMAGE_SIZE,
        NUM_WORKERS
    )

    num_classes = len(classes)
    print(f"Number of bird species: {num_classes}")
    print(f"Species found: {classes[:10]}... (showing first 10)")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # ============================================================
    # CREATE MODEL
    # ============================================================
    print("\nCreating model...")
    model = get_model(num_classes=num_classes).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============================================================
    # TRAIN MODEL
    # ============================================================
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    start_time = time.time()

    model, history = train_model(
        model,              # The model to train
        train_loader,       # Training data
        val_loader,         # Validation data
        device,             # GPU or CPU
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE
    )

    # Print training time
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.1f} minutes")

    # ============================================================
    # EVALUATE ON TEST SET
    # ============================================================
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)

    # Set model to evaluation mode
    model.eval()

    test_correct = 0
    test_total = 0

    # Disable gradients for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Correct: {test_correct}/{test_total}")

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================
    # Save model with class names (needed for inference later)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes,         # List of species names
        'test_acc': test_acc,       # Final test accuracy
        'history': history,         # Training history (for plotting)
    }, 'checkpoints/final_model.pth')

    print("\nFinal model saved to checkpoints/final_model.pth")
    print("Training complete! Use main.ipynb for inference.")


# ============================================================
# RUN TRAINING
# ============================================================
# This block runs when you execute: python train.py
if __name__ == "__main__":
    main()
