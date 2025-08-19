import torch
import torch.nn as nn
import torch.optim as optim
from SimpleViT import SimpleViT
from utils import *
from train import *
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _get_cifar10_dataloaders(
        samples_per_batch: int,
        train_validation_split: float
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the CIFAR-10 dataset and creates DataLoader objects for training,
    validation, and testing.

    Args:
        samples_per_batch (int): The batch size for DataLoaders.
        train_validation_split (float): Fraction of training data for validation.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # Transform: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 datasets
    full_train_dataset = datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

    # Split training dataset (train + validation)
    train_dataset, val_dataset = random_split(full_train_dataset,
                                              [1 - train_validation_split,
                                               train_validation_split])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=samples_per_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=samples_per_batch, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=samples_per_batch, shuffle=False)

    return train_loader, val_loader, test_loader


def _setup_model_for_training(
        num_classes: int,
        lr: float
) -> tuple[nn.Module, nn.modules.loss, torch.optim.Optimizer, torch.device]:

    # Set device (GPU/CPU)
    device = get_device()

    # Instantiate the SimpleCNN model
    model = SimpleViT(num_classes=num_classes).to(device)

    # Initialize the Cross-Entropy Loss function
    loss_function = nn.CrossEntropyLoss().to(device)

    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, loss_function, optimizer, device

def main():
    # Initialize model, loss function and optimizer
    vit, loss_fn, optimizer, device = _setup_model_for_training(10, 1e-4)

    train_loader, val_loader, test_loader = _get_cifar10_dataloaders(
        32, 0.2)

    # Train & Validation
    loss_records = train_model(
        model=vit,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        num_epochs=10)

    # Test
    test_accuracy, test_loss = evaluate_model(vit, test_loader, loss_fn, device)
    print(f"\nTest Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}%")

    # Plot Loss
    plot_losses(loss_records)