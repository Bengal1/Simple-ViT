import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from typing import  Sequence


__all__ = [
    "evaluate_model",
    "train_model"
]


def evaluate_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.modules.loss,
                   device: torch.device) -> float:

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for src, trg in data_loader:
            src, trg = src.to(device), trg.to(device)
            # Forward pass
            output = model(src, trg[:, :-1])

            # Flatten the tensors for loss computation
            logits = output.view(-1, output.size(-1))
            targets = trg[:, 1:].contiguous().view(-1)

            # Compute loss
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def train_model(model: nn.Module,
                criterion: nn.modules.loss,
                optimizer: torch.optim,
                training_loader: DataLoader,
                validation_loader: DataLoader,
                device: torch.device,
                num_epochs: int) -> dict[str,list[float]]:

    loss_record = {'train': [], 'validation': []}

    for epoch in range(num_epochs):
        train_accuracy, train_loss = _train_epoch(model,
                                                  criterion,
                                                  optimizer,
                                                  training_loader,
                                                  device)
        loss_record['train'].append(train_loss)

        validation_accuracy, validation_loss = evaluate_model(model,
                                                              criterion,
                                                              validation_loader,
                                                              device)
        loss_record['validation'].append(validation_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}% | Validation Loss:"
              f" {validation_loss:.4f}, Validation Accuracy:"
              f" {validation_accuracy:.2f}%")

        if _early_stopping(loss_record['validation'], 5):
            break

    return loss_record

 #--- Training Helper Functions ---
def _train_epoch(model: nn.Module,
                criterion: nn.modules.loss,
                optimizer: torch.optim,
                data_loader: DataLoader,
                device: torch.device) -> tuple[float, float]:

    model.train()  # Training mode
    correct_train, total_train, total_train_loss = 0, 0, 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Training accuracy calculation
        total_train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train += predicted.eq(labels).sum().item()
        total_train += labels.size(0)

    epoch_accuracy = 100 * correct_train / total_train
    epoch_loss = total_train_loss / len(data_loader)

    return epoch_accuracy, epoch_loss


def _early_stopping(metric_record: Sequence[float], patience: int = 5) -> bool:
    """
    Checks if a metric has improved in the last `patience` epochs.

    Args:
        metric_record: List/Sequence of metric values (e.g., BLEU or validation loss).
        patience: Number of epochs to wait for improvement.

    Returns:
        should_stop (bool): True if metric did not improve in last `patience` epochs.

    Raises:
        ValueError: If `patience` is not a positive integer.
    """
    if patience <= 0:
        raise ValueError("patience must be a positive integer")

    if len(metric_record) < patience + 1:
        return False  # not enough history yet

    # Best so far until "patience" epochs ago
    best_metric = max(metric_record[:-patience])

    # Check if last `patience` epochs improved
    recent_metrics = metric_record[-patience:]
    if max(recent_metrics) <= best_metric:
        return True

    return False