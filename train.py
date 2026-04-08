import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Sequence


# --- Public API ---
__all__ = [
    "evaluate_model",
    "train_model"
]


def evaluate_model(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.modules.loss,
        device: torch.device) -> tuple[float, float]:
    """
    Evaluates the model on a validation or test set.

    Args:
        model (nn.Module): The neural network model.
        data_loader (DataLoader): DataLoader for validation/test data.
        criterion (nn.modules.loss._Loss): The loss function.
        device (torch.device): The computational device (CPU or GPU).

    Returns:
        tuple[float, float]: Tuple containing evaluation accuracy (%) and average loss.
    """
    model.eval()  # Evaluation mode
    total_eval_loss, correct_eval, total_eval = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            # Accuracy calculation
            total_eval_loss += loss.item()
            _, predicted = logits.max(1)
            correct_eval += predicted.eq(labels).sum().item()
            total_eval += labels.size(0)

    eval_accuracy = 100 * correct_eval / total_eval
    eval_loss = total_eval_loss / len(data_loader)

    return eval_accuracy, eval_loss


def train_model(
        model: nn.Module,
        loss_fn: nn.modules.loss,
        optimizer: torch.optim,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        device: torch.device,
        num_epochs: int = 10,
        accumulation_steps: int = 1,
        max_gradient_clip: float | None = None,
        patience: int = 5
) -> dict[str, list[float]]:
    """
    Train a model for multiple epochs with validation and optional early stopping.

    Performs training using mini-batches, supports gradient accumulation and
    optional gradient clipping, and evaluates the model on a validation set
    after each epoch. Training may terminate early if the validation loss does
    not improve within a given patience.

    Args:
        model (nn.Module): The neural network model to train.
        loss_fn (nn.modules.loss): Loss function used for optimization.
        optimizer (torch.optim): Optimizer for updating model parameters.
        training_loader (DataLoader): DataLoader for the training dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform training on (CPU or GPU).
        num_epochs (int, optional): Maximum number of training epochs. Default is 10.
        accumulation_steps (int, optional): Number of batches to accumulate
            gradients before performing an optimizer step. Default is 1.
        max_gradient_clip (float | None, optional): Maximum gradient norm for
            clipping. If None, gradient clipping is disabled.
        patience (int, optional): Number of epochs to wait for validation loss
            improvement before triggering early stopping. Default is 5.

    Returns:
        dict[str, list[float]]:
            Dictionary containing per-epoch losses:
            - 'train': Training loss history
            - 'validation': Validation loss history

    Raises:
        ValueError: If ``accumulation_steps`` is not a positive integer.
        ValueError: If ``patience`` is not a positive integer.
    """
    loss_record = {'train': [], 'validation': []}

    for epoch in range(1, num_epochs + 1):
        train_accuracy, train_loss = _train_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_loader=training_loader,
            device=device,
            accumulation_steps=accumulation_steps,
            max_gradient_clip=max_gradient_clip
        )
        loss_record['train'].append(train_loss)

        validation_accuracy, validation_loss = evaluate_model(
            model=model,
            data_loader=validation_loader,
            criterion=loss_fn,
            device=device
        )
        loss_record['validation'].append(validation_loss)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}% | Validation Loss:"
              f" {validation_loss:.4f}, Validation Accuracy:"
              f" {validation_accuracy:.2f}%")

        if _early_stopping(
                metric_record=loss_record['validation'],
                patience=patience,
                best_is_max=False
        ):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return loss_record


# --- Training Helper Functions ---
def _train_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    accumulation_steps: int = 1,
    max_gradient_clip: float | None = None
) -> tuple[float, float]:
    """
    Perform a single training epoch.

    Executes forward and backward passes over the dataset, supports gradient
    accumulation to simulate larger batch sizes, and optionally applies gradient
    clipping before optimizer updates.

    Args:
        model (nn.Module): Model to train.
        loss_fn (nn.Module): Loss function used for optimization.
        optimizer (torch.optim.Optimizer): Optimizer.
        data_loader (DataLoader): Training data loader.
        device (torch.device): Device to perform training on.
        accumulation_steps (int, optional): Number of batches to accumulate
            gradients before performing an optimizer step. Default is 1.
        max_gradient_clip (float | None, optional): Maximum gradient norm for
            clipping. If None, gradient clipping is disabled.

    Returns:
        tuple[float, float]:
            - epoch_accuracy (float): Training accuracy in percent.
            - epoch_loss (float): Mean batch loss over the epoch.

    Raises:
        ValueError: If ``accumulation_steps`` is not a positive integer.
    """
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer")

    model.train()
    correct_train, total_train = 0, 0
    total_train_loss = 0.0

    optimizer.zero_grad()

    batch_idx = -1
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        total_train_loss += loss.item()

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if max_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_clip)

            optimizer.step()
            optimizer.zero_grad()

        _, predicted = logits.max(1)
        correct_train += predicted.eq(labels).sum().item()
        total_train += labels.size(0)

    if (batch_idx + 1) % accumulation_steps != 0:
        if max_gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_clip)

        optimizer.step()
        optimizer.zero_grad()

    epoch_accuracy = 100 * correct_train / total_train
    epoch_loss = total_train_loss / len(data_loader)

    return epoch_accuracy, epoch_loss


def _early_stopping(
    metric_record: Sequence[float],
    patience: int = 5,
    delta: float = 1e-5,
    best_is_max: bool = True
) -> bool:
    """
    Determine whether training should stop early based on a metric's recent performance.

    This function checks if the monitored metric has failed to improve
    within the last `patience` epochs, considering a minimum improvement
    threshold `delta`.

    Args:
        metric_record (Sequence[float]): Sequence of metric values
            (e.g., BLEU score or validation loss).
        patience (int, optional): Number of epochs to wait for improvement
            before suggesting early stopping. Must be positive. Default is 5.
        delta (float, optional): Minimum change in the metric to qualify as an
            improvement. Default is 1e-5.
        best_is_max (bool, optional): If True, higher metric values are better
            (e.g., BLEU). If False, lower metric values are better (e.g., loss).
            Default is True.

    Returns:
        bool: True if the metric did not improve sufficiently within the
            last `patience` epochs, indicating that training should stop.

    Raises:
        ValueError: If `patience` is not a positive integer.
    """
    if patience <= 0:
        raise ValueError("patience must be a positive integer")

    if len(metric_record) <= patience:
        return False  # not enough history yet

    if best_is_max:
        best_so_far = max(metric_record[:-patience])
        recent_best = max(metric_record[-patience:])
        return recent_best <= best_so_far - delta
    else:
        best_so_far = min(metric_record[:-patience])
        recent_best = min(metric_record[-patience:])
        return recent_best >= best_so_far + delta


# def _train_epoch(
#         model: nn.Module,
#         loss_fn: nn.modules.loss,
#         optimizer: torch.optim,
#         data_loader: DataLoader,
#         device: torch.device,
#         accumulation_steps: int = 1
# ) -> tuple[float, float]:
#     """
#     Perform a single training epoch on the given model using the provided
#     data loader.
#
#     Args:
#         model (nn.Module): The neural network model to train.
#         loss_fn (nn.modules.loss): Loss function used for training.
#         optimizer (torch.optim): Optimizer for updating model parameters.
#         data_loader (DataLoader): DataLoader providing batches of training data.
#         device (torch.device): Device to perform training on (CPU or GPU).
#         accumulation_steps (int): Number of batches to accumulate gradients
#                                 over before updating.
#
#     Returns:
#         tuple[float, float]: A tuple containing:
#             - epoch_accuracy (float): Training accuracy for this epoch (percent).
#             - epoch_loss (float): Average training loss over the epoch.
#     """
#     model.train()  # Training mode
#     correct_train, total_train, total_train_loss = 0, 0, 0
#
#     for images, labels in data_loader:
#         images, labels = images.to(device), labels.to(device)
#         # Reset gradients
#         optimizer.zero_grad()
#
#         # Forward pass
#         logits = model(images)
#
#         # Compute loss
#         loss = loss_fn(logits, labels)
#
#         # Backpropagation
#         loss.backward()
#
#         # Update parameters
#         optimizer.step()
#
#         # Training accuracy calculation
#         total_train_loss += loss.item()
#         _, predicted = logits.max(1)
#         correct_train += predicted.eq(labels).sum().item()
#         total_train += labels.size(0)
#
#     epoch_accuracy = 100 * correct_train / total_train
#     epoch_loss = total_train_loss / len(data_loader)
#
#     return epoch_accuracy, epoch_loss