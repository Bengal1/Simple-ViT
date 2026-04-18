# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
Main training entry point.

Orchestrates the full training pipeline:
    - Parses CLI arguments (dataset, model)
    - Builds dataloaders
    - Initializes model, loss, optimizer, and device
    - Runs training and validation
    - Evaluates on the test set
    - Saves metrics and generates plots

Configuration is managed via a centralized dataclass (`config`)
and can be partially overridden through CLI arguments.
"""
__author__="Bengal1"


import argparse

from config import config as cfg
from loaders import get_dataloaders
from utils import set_seed, plot_metrics, save_metrics_to_csv
from train import train_model, evaluate_model, setup_model_for_training


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for dataset and model selection.

    Returns:
        argparse.Namespace:
            Parsed command-line arguments.
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

    return parser.parse_args(args)

from utils import count_parameters

# --- Main Function ---
def main(args=None):
    set_seed()

    if args is None:
        args = parse_args()
    cfg.update_from_args(args)

    # Initialize data loaders
    train_loader, val_loader, test_loader, img_size, num_classes = get_dataloaders(
        dataset=cfg.dataset,
        batch_size=cfg.training.batch_size,
        train_validation_split=cfg.training.validation_split
    )

    # Initialize model, loss function and optimizer
    model, loss_fn, optimizer, device = setup_model_for_training(
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
        num_epochs=cfg.training.epochs
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
    save_metrics_to_csv(
        metrics_record=metrics_records,
        model_name=cfg.model_name,
        dataset=cfg.dataset,
        test_loss=test_loss,
        test_acc=test_accuracy
    )

    plot_metrics(
        statistics=metrics_records,
        model_name=cfg.model_name,
        dataset=cfg.dataset
    )


# --- Main Entry Point ---
if __name__ == "__main__":
    main()