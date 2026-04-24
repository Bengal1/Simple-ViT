import os
from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config as cfg
from utils import save_checkpoint

# ============================================================
# Public API
# ============================================================

__all__ = [
    "evaluate_model",
    "train_model",
]


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on a validation or test set.

    Args:
        model (nn.Module):
            Neural network model.
        data_loader (DataLoader):
            DataLoader for evaluation data.
        criterion (nn.Module):
            Loss function.
        device (torch.device):
            Computation device.

    Returns:
        tuple[float, float]:
            Evaluation accuracy (%) and average loss.
    """
    model.eval()

    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)

    return accuracy, avg_loss


# ============================================================
# Training
# ============================================================

def train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    accumulation_steps: int = 1,
    max_gradient_clip: float | None = None,
    patience: int = 5,
) -> dict[str, list[float]]:
    """
    Train a model with validation and optional early stopping.

    Runs multiple training epochs, evaluates on a validation set after each
    epoch, and tracks loss and accuracy. Training stops early if the
    validation metric does not improve.

    Args:
        model (nn.Module):
            Model to train.
        loss_fn (nn.Module):
            Loss function.
        optimizer (torch.optim.Optimizer):
            Optimizer.
        training_loader (DataLoader):
            Training data loader.
        validation_loader (DataLoader):
            Validation data loader.
        device (torch.device):
            Computation device.
        num_epochs (int, optional):
            Number of epochs.
        accumulation_steps (int, optional):
            Gradient accumulation steps.
        max_gradient_clip (float | None, optional):
            Max gradient norm for clipping.
        patience (int, optional):
            Early stopping patience.

    Returns:
        dict[str, list[float]]:
            Training statistics with keys:
            'train_loss', 'val_loss', 'train_acc', 'val_acc'.
    """
    stats = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_acc, train_loss = _train_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_loader=training_loader,
            device=device,
            accumulation_steps=accumulation_steps,
            max_gradient_clip=max_gradient_clip,
        )

        stats["train_loss"].append(train_loss)
        stats["train_acc"].append(train_acc)

        val_acc, val_loss = evaluate_model(
            model=model,
            data_loader=validation_loader,
            criterion=loss_fn,
            device=device,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, cfg.best_checkpoint_path)

        stats["val_loss"].append(val_loss)
        stats["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if _early_stopping(
            metric_record=stats["val_loss"],
            patience=patience,
            best_is_max=False,
        ):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return stats


# ============================================================
# Training Helpers
# ============================================================

def _train_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    accumulation_steps: int = 1,
    max_gradient_clip: float | None = None,
) -> tuple[float, float]:
    """
    Run a single training epoch.

    Performs forward and backward passes over the dataset, supports gradient
    accumulation, and optionally applies gradient clipping.

    Args:
        model (nn.Module):
            Model to train.
        loss_fn (nn.Module):
            Loss function.
        optimizer (torch.optim.Optimizer):
            Optimizer.
        data_loader (DataLoader):
            Training data loader.
        device (torch.device):
            Computation device.
        accumulation_steps (int, optional):
            Gradient accumulation steps.
        max_gradient_clip (float | None, optional):
            Max gradient norm for clipping.

    Returns:
        tuple[float, float]:
            Accuracy (%) and average loss.
    """
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")

    model.train()

    correct, total = 0, 0
    total_loss = 0.0

    optimizer.zero_grad()

    batch_idx = -1
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        total_loss += loss.item()

        (loss / accumulation_steps).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if max_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_gradient_clip
                )

            optimizer.step()
            optimizer.zero_grad()

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    if (batch_idx + 1) % accumulation_steps != 0:
        if max_gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_gradient_clip
            )

        optimizer.step()
        optimizer.zero_grad()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)

    return accuracy, avg_loss


def _early_stopping(
    metric_record: Sequence[float],
    patience: int = 5,
    delta: float = 1e-5,
    best_is_max: bool = True,
) -> bool:
    """
    Check whether training should stop early.

    Determines if the monitored metric has not improved within the last
    `patience` epochs, given a minimum improvement threshold.

    Args:
        metric_record (Sequence[float]):
            History of metric values.
        patience (int, optional):
            Number of epochs to wait for improvement.
        delta (float, optional):
            Minimum change to qualify as improvement.
        best_is_max (bool, optional):
            Whether higher values indicate improvement.

    Returns:
        bool:
            True if early stopping should be triggered.
    """
    if patience <= 0:
        raise ValueError("patience must be positive")

    if len(metric_record) <= patience:
        return False

    if best_is_max:
        best = max(metric_record[:-patience])
        recent = max(metric_record[-patience:])
        return recent <= best - delta

    best = min(metric_record[:-patience])
    recent = min(metric_record[-patience:])
    return recent >= best + delta