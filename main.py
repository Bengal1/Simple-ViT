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
from models.SimpleViT import SimpleViT
from utils import *
from train import *
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# # ----------------------- Hyperparameters & Config ----------------------- #
# # --- Model Architecture ---
EMBED_DIM        = 512       # Embedding dimension
NUM_HEADS        = 8         # Number of attention heads
NUM_LAYERS       = 6         # Number of Encoder/Decoder layers
PATCH_SIZE       = 4 #(16, 16)
# # --- Training Process ---
BATCH_SIZE       = 32        # Batch size
EPOCHS           = 10        # Number of epochs
NUM_CLASSES      = 10
VALIDATION_SPLIT = 0.2
# MAX_GRAD_CLIP   = 1.0       # Max norm gradient (for gradient clipping)
DROPOUT         = 0.1       # Dropout probability
# LABEL_SMOOTHING = 0.1       # Label smoothing parameter
# # --- Optimizer Settings (Adam) ---
LEARNING_RATE   = 1e-5      # Initial learning rate
# BETAS           = (0.9, 0.98) # Adam Optimizer beta coefficients
# EPSILON         = 1e-9      # Optimizer's epsilon for numerical stability
WEIGHT_DECAY    = 1e-2      # Weight decay parameter (L2 regularization)
# # --- Application-Specific Settings ---
# DATA_DEBUG_MODE = True      # Debug mode flag (enables/disables debug features)
# LOGGING_LEVEL   = utils.LogLevel.WARNING # Initial logging verbosity level


# --- Data Loader Initialization Helper Function ---
def _get_cifar10_dataloaders(
        samples_per_batch: int,
        train_validation_split: float
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    tuple[int, int, int]]:
    """
    Loads the CIFAR-10 dataset and creates DataLoader objects for training,
    validation, and testing.

    Args:
        samples_per_batch (int): The batch size for DataLoaders.
        train_validation_split (float): Fraction of training data for validation.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader, tuple]:
                                (train_loader, val_loader, test_loader, image_size)
    """
    # Transform: convert to tensor and normalize
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                          (0.2023, 0.1994, 0.2010))
    # ])

    # Load CIFAR-10 datasets
    # full_train_dataset = datasets.CIFAR10(root='./data', train=True,
    #                                       download=True, transform=transform)
    # test_dataset = datasets.CIFAR10(root='./data', train=False,
    #                                 download=True, transform=transform)

    full_train_dataset = datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transforms.ToTensor())
    # Split training dataset (train + validation)
    train_dataset, val_dataset = random_split(full_train_dataset,
                                              [1 - train_validation_split,
                                               train_validation_split])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=samples_per_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=samples_per_batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=samples_per_batch, shuffle=False)

    # Get image size
    image_size = next(iter(train_loader))[0].shape[1:]

    return train_loader, val_loader, test_loader, image_size


# --- Model & training Component Setup Helper Function ---
def _setup_model_for_training(
        num_classes: int,
        patch_size: int | tuple[int, int],
        lr: float,
        img_size: int | tuple[int, int, int] | None = None,
) -> tuple[
    nn.Module,
    nn.modules.loss,
    torch.optim.Optimizer,
    torch.device]:
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

    # Instantiate the SimpleCNN model
    model = SimpleViT(patch_size=patch_size,
                      num_classes=num_classes,
                      img_size=img_size,
                      dropout=DROPOUT
                      ).to(device)

    # Initialize the Cross-Entropy Loss function
    loss_function = nn.CrossEntropyLoss().to(device)

    # Initialize the Adam optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    return model, loss_function, optimizer, device


# --- Main Function ---
def main():
    set_seed()

    # Initialize data loaders
    train_loader, val_loader, test_loader, img_size = _get_cifar10_dataloaders(
        BATCH_SIZE, VALIDATION_SPLIT)

    # Initialize model, loss function and optimizer
    vit, loss_fn, optimizer, device = _setup_model_for_training(NUM_CLASSES,
                                                                PATCH_SIZE,
                                                                LEARNING_RATE,
                                                                img_size)

    # Train & Validation
    loss_records = train_model(
        model=vit,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        num_epochs=10)

    # Test
    test_accuracy, test_loss = evaluate_model(vit, test_loader, loss_fn, device)
    print(f"\nTest Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}%")

    # Plot Loss
    plot_losses(loss_records)


# --- Main Entry Point ---
if __name__ == "__main__":
    main()