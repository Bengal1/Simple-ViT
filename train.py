import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from typing import  Sequence


__all__ = [
    "evaluate_model",
    "train_model"
]


def evaluate_model(model: nn.Module,
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


def train_model(model: nn.Module,
                loss_fn: nn.modules.loss,
                optimizer: torch.optim,
                training_loader: DataLoader,
                validation_loader: DataLoader,
                device: torch.device,
                num_epochs: int) -> dict[str,list[float]]:

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

        if _early_stopping(loss_record['validation'], 5):
            break

    return loss_record


 #--- Training Helper Functions ---
def _train_epoch(model: nn.Module,
                loss_fn: nn.modules.loss,
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
        loss = loss_fn(outputs, labels)

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