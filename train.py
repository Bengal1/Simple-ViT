import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import  Sequence


__all__ = [
    "evaluate_model",
    "train_model"
]


def evaluate_model(
        model: nn.Module,
        criterion: nn.modules.loss,
        data_loader: DataLoader,
        device: torch.device) -> tuple[float, float]:
    """
    Evaluates the model on a validation or test set.

    Args:
        model (nn.Module): The neural network model.
        criterion (nn.modules.loss._Loss): The loss function.
        data_loader (DataLoader): DataLoader for validation/test data.
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
                                                              loss_fn,
                                                              validation_loader,
                                                              device)
        loss_record['validation'].append(validation_loss)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}% | Validation Loss:"
              f" {validation_loss:.4f}, Validation Accuracy:"
              f" {validation_accuracy:.2f}%")

        if _early_stopping(loss_record['validation'], patience=5, best_is_max=False):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return loss_record


 #--- Training Helper Functions ---
def _train_epoch(
        model: nn.Module,
        loss_fn: nn.modules.loss,
        optimizer: torch.optim,
        data_loader: DataLoader,
        device: torch.device) -> tuple[float, float]:
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


def _early_stopping(
        metric_record: Sequence[float],
        patience: int = 5,
        best_is_max: bool = True) -> bool:
    """
    Checks if a metric has failed to improve within the last `patience`
    epochs.

    Args:
        metric_record: Sequence of metric values (e.g., BLEU or validation loss).
        patience: Number of epochs to wait for improvement.
        best_is_max: Whether higher metric values are better (like BLEU)
            or lower (like loss).

    Returns:
        should_stop (bool): True if metric did not improve in the last
            `patience` epochs.

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
        return recent_best <= best_so_far
    else:
        best_so_far = min(metric_record[:-patience])
        recent_best = min(metric_record[-patience:])
        return recent_best >= best_so_far