import os
import csv
import torch
import random
import logging
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


# --- Public API ---
__all__ = [
    "get_device",
    "set_seed",
    "plot_metrics",
    "save_metrics_to_csv"
]


# -------------- Device Configuration --------------- #

def get_device() -> torch.device:
    """
    Selects and returns the optimal device (GPU or CPU) for computation.

    This function first checks for the availability of a NVIDIA GPU with
    CUDA support. If a GPU is found, it's chosen as the computation device.
    Otherwise, it defaults to the CPU. A descriptive message is printed to
    inform the user which device has been selected. This helps in verifying
    that the hardware is correctly recognized for accelerated computations.

    Returns:
        torch.device: The selected device, either 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device)
        print(f"Using GPU: {device_name}\n")
    else:
        device = torch.device('cpu')
        print("Using CPU\n")

    return device


def set_seed(seed_value: int = 1755900008) -> None:
    """
    Sets the random seed for reproducibility across multiple libraries.

    This function ensures that the random number generators in Python's
    built-in `random` module, NumPy, and PyTorch are all initialized
    with the same seed. This is crucial for creating reproducible
    experiments in machine learning, as it guarantees that operations
    involving randomness (like data shuffling, weight initialization,
    and dropout) will yield the same results every time the code is run.

    Args:
        seed_value (int): The integer value to use as the seed. Defaults to 1755900008.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # If a GPU is available, set the seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


# ------------------ Metrics ------------------ #
def plot_metrics(
    statistics: dict[str, list[float]],
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    save_dir: str = "results",
    mode: str = "combined",
) -> None:
    """
    Plot training metrics and save the resulting figure.

    This function visualizes training progress using loss and accuracy
    curves, and optionally analyzes generalization through the loss gap.

    Visualization Modes:
        - "combined" : Two subplots (Loss and Accuracy).
        - "loss"     : Single plot of training and validation loss.
        - "accuracy" : Single plot of training and validation accuracy.
        - "gap"      : Two subplots:
                          1. Loss gap (train - validation)
                          2. Loss curves with shaded gap area
        - "extended" : Three subplots:
                          1. Loss
                          2. Accuracy
                          3. Loss gap

    If both `model_name` and `dataset` are provided, the figure is saved as:
        {model_name}_{dataset}_{mode}_metrics.png

    Otherwise, the figure is saved using the next available filename:
        metrics_1.png, metrics_2.png, ...

    Args:
        statistics (dict[str, list[float]]):
            Dictionary containing per-epoch metrics with the following keys:
                - "train_loss"
                - "val_loss"
                - "train_acc"
                - "val_acc"
        model_name (Optional[str], optional):
            Model name used in the saved figure filename.
            Defaults to None.
        dataset (Optional[str], optional):
            Dataset name used in the saved figure filename.
            Defaults to None.
        save_dir (str, optional):
            Directory in which to save the generated figure.
            Defaults to "results".
        mode (str, optional):
            Visualization mode. Must be one of:
            {"combined", "loss", "accuracy", "gap", "extended"}.
            Defaults to "combined".

    Raises:
        ValueError:
            If required keys are missing from `statistics`, if metric lists
            do not share the same length, or if `mode` is invalid.
    """
    required_keys = {"train_loss", "val_loss", "train_acc", "val_acc"}
    if not required_keys.issubset(statistics):
        raise ValueError(f"statistics must contain keys: {required_keys}")

    train_loss = statistics["train_loss"]
    val_loss = statistics["val_loss"]
    train_acc = statistics["train_acc"]
    val_acc = statistics["val_acc"]

    lengths = {len(train_loss), len(val_loss), len(train_acc), len(val_acc)}
    if len(lengths) != 1:
        raise ValueError("All metric lists in `statistics` must have the same length.")

    epochs = range(1, len(train_loss) + 1)

    os.makedirs(save_dir, exist_ok=True)
    save_path = _build_metrics_save_path(
        save_dir=save_dir,
        model_name=model_name,
        dataset=dataset,
        mode=mode,
    )

    fig = _plot_metrics_by_mode(
        mode=mode,
        epochs=epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc,
    )

    if model_name is not None and dataset is not None:
        fig.suptitle(
            f"{model_name.upper()} | {dataset.upper()}",
            fontsize=18,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def _build_metrics_save_path(
    save_dir: str,
    model_name: Optional[str],
    dataset: Optional[str],
    mode: str,
) -> str:
    """
    Build the output path for a metrics figure.

    If both `model_name` and `dataset` are provided, the filename format is:
        {model_name}_{dataset}_{mode}_metrics.png

    Otherwise, the function generates the next available sequential filename:
        metrics_1.png, metrics_2.png, ...

    Args:
        save_dir (str):
            Directory in which the figure will be saved.
        model_name (Optional[str]):
            Model name for the filename.
        dataset (Optional[str]):
            Dataset name for the filename.
        mode (str):
            Plot mode included in the filename when names are provided.

    Returns:
        str:
            Full path to the output figure file.
    """
    if model_name is not None and dataset is not None:
        file_name = f"{model_name.lower()}_{dataset.lower()}_{mode}_metrics.png"
        return os.path.join(save_dir, file_name)

    file_index = 1
    while True:
        file_name = f"metrics_{file_index}.png"
        save_path = os.path.join(save_dir, file_name)
        if not os.path.exists(save_path):
            return save_path
        file_index += 1


def _plot_metrics_by_mode(
    mode: str,
    epochs: range,
    train_loss: list[float],
    val_loss: list[float],
    train_acc: list[float],
    val_acc: list[float],
) -> plt.Figure:
    """
    Plot metrics according to the selected visualization mode.

    Args:
        mode (str):
            Visualization mode. Supported values:
            {"combined", "loss", "accuracy", "gap", "extended"}.
        epochs (range):
            Epoch indices used for the x-axis.
        train_loss (list[float]):
            Training loss values.
        val_loss (list[float]):
            Validation loss values.
        train_acc (list[float]):
            Training accuracy values.
        val_acc (list[float]):
            Validation accuracy values.

    Returns:
    plt.Figure: The created matplotlib figure.

    Raises:
        ValueError:
            If `mode` is not one of the supported options.
    """
    loss_gap = [train - val for train, val in zip(train_loss, val_loss)]

    if mode == "combined":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
        axes[0].plot(epochs, val_loss, label="Validation Loss", linewidth=2)
        axes[0].set_title("Loss", fontsize=16, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        axes[1].plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
        axes[1].plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
        axes[1].set_title("Accuracy", fontsize=16, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()

        return fig

    elif mode == "loss":
        fig = plt.figure(figsize=(7, 5), dpi=150)

        plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
        plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
        plt.title("Loss", fontsize=16, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        return fig

    elif mode == "accuracy":
        fig = plt.figure(figsize=(7, 5), dpi=150)

        plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
        plt.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
        plt.title("Accuracy", fontsize=16, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        return fig

    elif mode == "gap":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        axes[0].plot(epochs, loss_gap, label="Loss Gap", linewidth=2)
        axes[0].axhline(0, linestyle="--", linewidth=1)
        axes[0].set_title("Loss Gap (Train - Val)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Gap")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        axes[1].plot(epochs, train_loss, label="Train Loss", linewidth=2)
        axes[1].plot(epochs, val_loss, label="Validation Loss", linewidth=2)
        axes[1].fill_between(
            epochs,
            train_loss,
            val_loss,
            alpha=0.2,
            label="Gap Area",
        )
        axes[1].set_title("Loss with Gap Area", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()

        return fig

    elif mode == "extended":
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)

        axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
        axes[0].plot(epochs, val_loss, label="Validation Loss", linewidth=2)
        axes[0].set_title("Loss", fontsize=16, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        axes[1].plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
        axes[1].plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
        axes[1].set_title("Accuracy", fontsize=16, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()

        axes[2].plot(epochs, loss_gap, label="Loss Gap", linewidth=2)
        axes[2].axhline(0, linestyle="--", linewidth=1)
        axes[2].set_title("Loss Gap (Train - Val)", fontsize=16, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Gap")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].legend()

        return fig

    else:
        raise ValueError(
            "mode must be one of: "
            "'combined', 'loss', 'accuracy', 'gap', 'extended'"
        )


def save_metrics_to_csv(
            metrics_record: dict[str, list[float]],
            model_name: str,
            dataset: str,
            save_dir: str = "results",
            test_loss: Optional[float] = None,
            test_acc: Optional[float] = None,
) -> None:
    """
    Save training metrics to a CSV file.

    Args:
        metrics_record (dict):
            Dictionary containing metric lists per epoch
            (e.g., 'train_loss', 'val_loss', 'train_acc', 'val_acc').
        model_name (str): Model name (e.g., 'cnn', 'vit').
        dataset (str): Dataset name (e.g., 'mnist', 'cifar10').
        save_dir (str): Directory to save the CSV file.
        test_loss (Optional[float]): Final test loss.
        test_acc (Optional[float]): Final test accuracy.
    """

    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f"{model_name}_{dataset}.csv")

    keys = list(metrics_record.keys())
    num_epochs = len(next(iter(metrics_record.values())))

    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        # header
        writer.writerow(["epoch"] + keys)

        # rows
        for i in range(num_epochs):
            row = [i + 1] + [metrics_record[k][i] for k in keys]
            writer.writerow(row)

    # ---- Test metrics block ----
    if test_loss is not None or test_acc is not None:
        writer.writerow([])  # empty line
        writer.writerow(["test_metrics"])

        if test_loss is not None:
            writer.writerow(["test_loss", test_loss])

        if test_acc is not None:
            writer.writerow(["test_accuracy", test_acc])

    print(f"Saved metrics to: {file_path}")
