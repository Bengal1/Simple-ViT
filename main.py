# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
__author__="Bengal1"

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from models import SimpleViT
from loaders import get_dataloaders
from train import train_model, evaluate_model
from utils import get_device, plot_losses, set_seed
from SimpleCNN import SimpleCNN


# # ----------------------- Hyperparameters & Config ----------------------- #
# # --- Model Architecture ---
EMBED_DIM        = 512       # Embedding dimension
NUM_HEADS        = 8         # Number of attention heads
NUM_LAYERS       = 6         # Number of Encoder/Decoder layers
PATCH_SIZE       = 4 #(16, 16)
# # --- Training Process ---
BATCH_SIZE       = 32        # Batch size
EPOCHS           = 100        # Number of epochs
NUM_CLASSES      = 10
VALIDATION_SPLIT = 0.2
MAX_GRAD_CLIP    = 1.0       # Max norm gradient (for gradient clipping)
DROPOUT          = 0.1       # Dropout probability
LABEL_SMOOTHING  = 0.1       # Label smoothing parameter
# # --- Optimizer Settings (Adam) ---
LEARNING_RATE    = 1e-5      # Initial learning rate
BETAS            = (0.9, 0.98) # Adam Optimizer beta coefficients
EPSILON          = 1e-9      # Optimizer's epsilon for numerical stability
WEIGHT_DECAY     = 1e-2      # Weight decay parameter (L2 regularization)
# # --- Application-Specific Settings ---
# DATA_DEBUG_MODE  = True      # Debug mode flag (enables/disables debug features)
# LOGGING_LEVEL    = utils.LogLevel.WARNING # Initial logging verbosity level

DATASET         = "mnist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classification models.")

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "tiny_imagenet"],
        help="Dataset to use for training.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "cnn"],
        help="Model architecture to train.",
    )

    return parser.parse_args()


# --- Model & training Component Setup Helper Function ---
def _setup_model_for_training(
        num_classes: int,
        patch_size: int | tuple[int, int],
        lr: float,
        img_size: int | tuple[int, int, int],
        model: str = "vit"
) -> tuple[
    nn.Module,
    nn.modules.loss,
    torch.optim.Optimizer,
    torch.device
]:
    """
    Set up a Simple Vision Transformer (SimpleViT) model for training.

    This function initializes the model, the loss function, the optimizer,
    and selects the appropriate device (CPU or GPU) for training.

    Args:
        num_classes (int): Number of output classes for classification.
        patch_size (int or tuple[int, int]): Size of each patch for the ViT.
        lr (float): Learning rate for the optimizer.
        img_size (tuple[int, int, int]): Input image size.

    Returns:
        tuple:
            - model (nn.Module): The instantiated SimpleViT model on the 
              selected device.
            - loss_function (nn.modules.loss): Cross-Entropy loss function 
              on the same device.
            - optimizer (torch.optim.Optimizer): AdamW optimizer for model 
              parameters.
            - device (torch.device): The device used for training (CPU or GPU).
    """
    # Set device (GPU/CPU)
    device = get_device()

    # Instantiate the SimpleViT model
    if model == "vit":
        model = SimpleViT(
            patch_size=patch_size,
            num_classes=num_classes,
            img_size=img_size,
            dropout=DROPOUT
        ).to(device)
    elif model == "cnn":
        model = SimpleCNN()

    # Initialize the Cross-Entropy Loss function
    loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(device)

    # Initialize the Adam optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY
    )

    return model, loss_function, optimizer, device


# --- Main Function ---
def main():
    set_seed()

    # Initialize data loaders
    train_loader, val_loader, test_loader, img_size = get_dataloaders(
        dataset="mnist",
        batch_size=BATCH_SIZE,
        train_validation_split=VALIDATION_SPLIT
    )

    # Initialize model, loss function and optimizer
    vit, loss_fn, optimizer, device = _setup_model_for_training(
        num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        lr=LEARNING_RATE,
        img_size=img_size
    )

    # Train & Validation
    loss_records = train_model(
        model=vit,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        num_epochs=EPOCHS
    )

    # Test
    test_accuracy, test_loss = evaluate_model(
        model=vit,
        data_loader=test_loader,
        criterion=loss_fn,
        device=device
    )
    print(f"\nTest Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}%")

    # Plot Loss
    plot_losses(loss_records)


# --- Main Entry Point ---
if __name__ == "__main__":
    main()