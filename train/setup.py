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
    Initialize model, loss function, optimizer, and device.

    Args:
        config (Config):
            Global configuration object.
        num_classes (int):
            Number of output classes.
        img_size (tuple[int, int, int]):
            Input image shape (C, H, W).
        model_name (str):
            Model type ("vit" or "cnn").

    Returns:
        tuple:
            model (nn.Module):
                Initialized model on target device.
            loss_fn (nn.Module):
                Cross-entropy loss function.
            optimizer (torch.optim.Optimizer):
                Configured optimizer.
            device (torch.device):
                Selected computation device.

    Raises:
        ValueError:
            If an unsupported model name is provided.
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
    Instantiate a model and move it to the target device.

    This function acts as a factory for supported model architectures,
    initializing the appropriate model with the given configuration
    and ensuring it is placed on the specified device.

    Args:
        model_name (str): Model identifier. Supported values: {"vit", "cnn"}.
        config (Config): Global configuration containing model-specific settings.
        num_classes (int): Number of output classes for the classification head.
        img_size (tuple[int, int, int]): Input image shape as (C, H, W).
        device (torch.device): Target device on which the model will be allocated.

    Returns:
        nn.Module:
            Instantiated model moved to the specified device.

    Raises:
        ValueError:
            If `model_name` is not one of the supported architectures.
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


# def setup_model_for_training(
#         config: Config,
#         num_classes: int,
#         img_size: int | tuple[int, int, int],
#         model_name: str
# ) -> tuple[
#     nn.Module,
#     nn.modules.loss,
#     torch.optim.Optimizer,
#     torch.device
# ]:
#     """
#     Initialize model, loss function, optimizer, and device.
#
#     Supports multiple model architectures (e.g., ViT, CNN) based on
#     the provided configuration and model name.
#
#     Args:
#         config (Config): Global configuration object.
#         num_classes (int): Number of output classes.
#         img_size (tuple[int, int, int]): Input image size (C, H, W).
#         model_name (str): Model type to instantiate ("vit" or "cnn").
#
#     Returns:
#         tuple:
#             - model (nn.Module): Initialized model on target device.
#             - loss_function (nn.Module): Cross-entropy loss function.
#             - optimizer (torch.optim.Optimizer): Configured optimizer.
#             - device (torch.device): Selected computation device.
#
#     Raises:
#         ValueError: If an unsupported model name is provided.
#     """
#     # Set device (GPU/CPU)
#     device = get_device()
#
#     # Instantiate the Selected model
#     if model_name == "vit":
#         model = SimpleViT(
#             cfg=config.vit,
#             num_classes=num_classes,
#             img_size=img_size,
#         ).to(device)
#     elif model_name == "cnn":
#         model = SimpleCNN(
#             input_shape=img_size,
#             num_classes=num_classes,
#             cfg=config.cnn
#         ).to(device)
#     else:
#         raise ValueError(f"Unsupported model '{model_name}'")
#
#
#     # Initialize the Cross-Entropy Loss function
#     loss_function = nn.CrossEntropyLoss(
#         label_smoothing=config.training.label_smooth
#     ).to(device)
#
#     # Initialize the Adam optimizer
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=config.optim.learning_rate,
#         betas=config.optim.betas,
#         eps=config.optim.eps,
#         weight_decay=config.optim.weight_decay
#     )
#
#     return model, loss_function, optimizer, device