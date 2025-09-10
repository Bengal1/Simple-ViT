import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import  Sequence


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
        num_epochs: int) -> dict[str,list[float]]:
    """
    Train a PyTorch model for a specified number of epochs while evaluating
    on a validation set after each epoch.

    Args:
        model (nn.Module): The neural network model to train.
        loss_fn (nn.modules.loss): Loss function used for training.
        optimizer (torch.optim): Optimizer for updating model parameters.
        training_loader (DataLoader): DataLoader for the training dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform training on (CPU or GPU).
        num_epochs (int): Maximum number of training epochs.

    Returns:
        dict[str, list[float]]: Dictionary containing lists of training and
        validation losses for each epoch with keys 'train' and 'validation'.
    """
    loss_record = {'train': [], 'validation': []}

    for epoch in range(1, num_epochs + 1):
        train_accuracy, train_loss = _train_epoch(model,
                                                  loss_fn,
                                                  optimizer,
                                                  training_loader,
                                                  device)
        loss_record['train'].append(train_loss)

        validation_accuracy, validation_loss = evaluate_model(model,
                                                              validation_loader,
                                                              loss_fn,
                                                              device)
        loss_record['validation'].append(validation_loss)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}% | Validation Loss:"
              f" {validation_loss:.4f}, Validation Accuracy:"
              f" {validation_accuracy:.2f}%")

        if _early_stopping(loss_record['validation'], patience=7, best_is_max=False):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return loss_record


 # --- Training Helper Functions ---
def _train_epoch(
        model: nn.Module,
        loss_fn: nn.modules.loss,
        optimizer: torch.optim,
        data_loader: DataLoader,
        device: torch.device
) -> tuple[float, float]:
    """
    Perform a single training epoch on the given model using the provided
    data loader.

    Args:
        model (nn.Module): The neural network model to train.
        loss_fn (nn.modules.loss): Loss function used for training.
        optimizer (torch.optim): Optimizer for updating model parameters.
        data_loader (DataLoader): DataLoader providing batches of training data.
        device (torch.device): Device to perform training on (CPU or GPU).

    Returns:
        tuple[float, float]: A tuple containing:
            - epoch_accuracy (float): Training accuracy for this epoch (percent).
            - epoch_loss (float): Average training loss over the epoch.
    """
    model.train()  # Training mode
    correct_train, total_train, total_train_loss = 0, 0, 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = loss_fn(logits, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Training accuracy calculation
        total_train_loss += loss.item()
        _, predicted = logits.max(1)
        correct_train += predicted.eq(labels).sum().item()
        total_train += labels.size(0)

    epoch_accuracy = 100 * correct_train / total_train
    epoch_loss = total_train_loss / len(data_loader)

    return epoch_accuracy, epoch_loss


# def _train_epoch(
#         model: torch.nn.Module,
#         train_loader: torch.utils.data.DataLoader,
#         optimizer: torch.optim.Optimizer,
#         scheduler: torch.optim.lr_scheduler._LRScheduler,
#         criterion: torch.nn.modules.loss,
#         device: torch.device,
#         max_gradient_clip: float,
#         target_vocab_size: int,
#         accumulation_steps: int = 1
# ) -> float:
#     """
#     Performs one epoch of training on the given model with gradient accumulation.
#
#     Gradient accumulation allows simulating a larger batch size by accumulating
#     gradients over multiple smaller batches before performing an optimizer step.
#
#     Args:
#         model (torch.nn.Module): The model to train.
#         train_loader (DataLoader): DataLoader for the training dataset.
#         optimizer (Optimizer): Optimizer used for updating model parameters.
#         scheduler (_LRScheduler): Learning rate scheduler.
#         criterion (_Loss): Loss function.
#         device (torch.device): Device to run the training on.
#         max_gradient_clip (float): Maximum gradient norm for clipping.
#         target_vocab_size (int): Size of the target vocabulary (for reshaping logits).
#         accumulation_steps (int): Number of batches to accumulate gradients over before updating.
#
#     Returns:
#         float: Average training loss over the epoch (per batch, not per accumulated step).
#     """
#     model.train()
#     running_loss = 0.0  # Sum of batch losses for reporting
#
#     optimizer.zero_grad()  # Reset gradients at the start of the epoch
#
#     batch_idx = -1
#     for batch_idx, (image_batch, label_batch) in enumerate(train_loader):
#         image_batch, label_batch = image_batch.to(device), label_batch.to(device)
#
#         # Forward pass
#         logits = model(image_batch)  # Teacher forcing
#
#         # Compute loss for current batch and scale by accumulation_steps
#         loss = criterion(logits, label_batch) / accumulation_steps
#         running_loss += (
#                     loss.item() * accumulation_steps)  # accumulate unscaled loss for reporting
#
#         # Backpropagate scaled loss
#         loss.backward()
#
#         # Perform optimizer step every `accumulation_steps` batches
#         if (batch_idx + 1) % accumulation_steps == 0:
#             # Clip gradients to avoid exploding gradients
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_clip)
#
#             # Update parameters
#             optimizer.step()
#             scheduler.step()
#
#             # Reset gradients after update
#             optimizer.zero_grad()
#
#     # Handle remaining gradients if number of batches is not divisible by accumulation_steps
#     if (batch_idx + 1) % accumulation_steps != 0:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_clip)
#         optimizer.step()
#         scheduler.step()
#         optimizer.zero_grad()
#
#     # Return average batch loss over the epoch
#     return running_loss / len(train_loader)


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
