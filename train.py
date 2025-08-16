import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


def _train_epoch(model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.modules.loss,
                 train_loader: DataLoader,
                 device: torch.device) -> float:

    total_train_loss = 0.0

    for imgs, labels in train_loader:
        imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = loss_fn(outputs,labels)
        total_train_loss += loss.item()

        loss.backward()

        optimizer.step()

    return total_train_loss / len(train_loader)

def train_model(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.modules.loss,
                train_loader: DataLoader,
                validation_loader: DataLoader, epochs: int,
                device: torch.device) -> dict[str, list[float]]:

    stats_record = {'train': [], 'validation': []}

    for epoch in range(1, epochs + 1):
        # Training
        train_loss = _train_epoch(model, optimizer, loss_fn, train_loader, device)
        stats_record['train'].append(train_loss)
        # Validation




    return stats_record