# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
Main training entry point.

Handles:
    - CLI argument parsing (dataset, model selection)
    - Data loading
    - Model initialization (ViT or CNN)
    - Training and validation loop
    - Final evaluation and loss visualization

The configuration is managed via a centralized dataclass-based system (`config`),
which can be partially overridden through CLI arguments.
"""
__author__="Bengal1"

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from models import SimpleViT, SimpleCNN
from loaders import get_dataloaders
from config import config as cfg, Config
from train import train_model, evaluate_model
from utils import get_device, set_seed, plot_metrics, save_metrics_to_csv




def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset and model selection.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
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
        config: Config,
        num_classes: int,
        img_size: int | tuple[int, int, int],
        model_name: str
) -> tuple[
    nn.Module,
    nn.modules.loss,
    torch.optim.Optimizer,
    torch.device
]:
    """
    Initialize model, loss function, optimizer, and device.

    Supports multiple model architectures (e.g., ViT, CNN) based on
    the provided configuration and model name.

    Args:
        config (Config): Global configuration object.
        num_classes (int): Number of output classes.
        img_size (tuple[int, int, int]): Input image size (C, H, W).
        model_name (str): Model type to instantiate ("vit" or "cnn").

    Returns:
        tuple:
            - model (nn.Module): Initialized model on target device.
            - loss_function (nn.Module): Cross-entropy loss function.
            - optimizer (torch.optim.Optimizer): Configured optimizer.
            - device (torch.device): Selected computation device.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    # Set device (GPU/CPU)
    device = get_device()

    # Instantiate the Selected model
    if model_name == "vit":
        model = SimpleViT(
            cfg=cfg.vit,
            num_classes=num_classes,
            img_size=img_size,
        ).to(device)
    elif model_name == "cnn":
        model = SimpleCNN(
            input_shape=img_size,
            num_classes=num_classes,
            cfg=cfg.cnn
        ).to(device)
    else:
        raise ValueError(f"Unsupported model '{model_name}'")


    # Initialize the Cross-Entropy Loss function
    loss_function = nn.CrossEntropyLoss(
        label_smoothing=cfg.training.label_smooth
    ).to(device)

    # Initialize the Adam optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optim.learning_rate,
        betas=config.optim.betas,
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay
    )

    return model, loss_function, optimizer, device


# --- Main Function ---
def main():
    set_seed()
    args = parse_args()
    cfg.dataset = args.dataset
    cfg.model_name = args.model

    # Initialize data loaders
    train_loader, val_loader, test_loader, img_size, num_classes = get_dataloaders(
        dataset=cfg.dataset,
        batch_size=cfg.training.batch_size,
        train_validation_split=cfg.training.validation_split
    )

    # Initialize model, loss function and optimizer
    model, loss_fn, optimizer, device = _setup_model_for_training(
        config=cfg,
        num_classes=num_classes,
        img_size=img_size,
        model_name=cfg.model_name
    )

    # Train & Validation
    metrics_records = train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        # num_epochs=cfg.training.epochs
    )

    # Test
    test_accuracy, test_loss = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=loss_fn,
        device=device
    )
    print(f"\nTest Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}%")

    # Save & Plot Metrics
    save_metrics_to_csv(metrics_records, cfg.model_name, cfg.dataset)
    plot_metrics(metrics_records,cfg.model_name, cfg.dataset)


# --- Main Entry Point ---
if __name__ == "__main__":
    main()