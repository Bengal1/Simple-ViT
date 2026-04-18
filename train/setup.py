import torch
import torch.nn as nn
import torch.optim as optim

from models import SimpleViT, SimpleCNN
from config import Config
from utils import get_device


def setup_model_for_training(
    config: Config,
    num_classes: int,
    img_size: tuple[int, int, int],
    model_name: str,
) -> tuple[nn.Module, nn.Module, optim.Optimizer, torch.device]:
    """
    Initialize the training components for a given model.

    Constructs the model, loss function, optimizer, and selects the
    appropriate computation device based on the provided configuration.

    Args:
        config (Config):
            Global configuration object containing model and training settings.
        num_classes (int):
            Number of output classes.
        img_size (tuple[int, int, int]):
            Input image shape as (C, H, W).
        model_name (str):
            Model type to instantiate ("vit" or "cnn").

    Returns:
        tuple[nn.Module, nn.Module, optim.Optimizer, torch.device]:
            Model, loss function, optimizer, and device.
    """
    device = get_device()

    model = _build_model(
        model_name=model_name,
        config=config,
        num_classes=num_classes,
        img_size=img_size,
        device=device,
    )

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=config.training.label_smooth,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.optim.learning_rate,
        betas=config.optim.betas,
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )

    return model, loss_fn, optimizer, device


def _build_model(
    model_name: str,
    config: Config,
    num_classes: int,
    img_size: tuple[int, int, int],
    device: torch.device,
) -> nn.Module:
    """
    Create the specified model and move it to the target device.

    Args:
        model_name (str):
            Model identifier ("vit" or "cnn").
        config (Config):
            Configuration containing model-specific parameters.
        num_classes (int):
            Number of output classes.
        img_size (tuple[int, int, int]):
            Input image shape as (C, H, W).
        device (torch.device):
            Target device.

    Returns:
        nn.Module:
            Instantiated model on the specified device.

    Raises:
        ValueError:
            If the model name is not supported.
    """
    if model_name == "vit":
        model = SimpleViT(
            cfg=config.vit,
            num_classes=num_classes,
            img_size=img_size,
        )
    elif model_name == "cnn":
        model = SimpleCNN(
            input_shape=img_size,
            num_classes=num_classes,
            cfg=config.cnn,
        )
    else:
        raise ValueError(f"Unsupported model '{model_name}'")

    return model.to(device)
