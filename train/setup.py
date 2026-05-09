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
) -> tuple[
    nn.Module,
    nn.Module,
    optim.Optimizer,
    torch.device,
    optim.lr_scheduler.LRScheduler | None,
]:
    """
    Initialize the training components for a given model.

    Constructs the model, loss function, optimizer, optional learning-rate
    scheduler, and selects the appropriate computation device.

    Args:
        config (Config):
            Global configuration object containing model and training settings.
        num_classes (int):
            Number of output classes.
        img_size (tuple[int, int, int]):
            Input image shape as (C, H, W).

    Returns:
        tuple[
            nn.Module,
            nn.Module,
            optim.Optimizer,
            torch.device,
            optim.lr_scheduler.LRScheduler | None,
        ]:
            Model, loss function, optimizer, computation device,
            and optional learning-rate scheduler.
    """
    device = get_device()

    model = _build_model(
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

    scheduler = None

    if config.training.use_scheduler:

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.training.warmup_start_factor,
            total_iters=config.training.warmup_epochs,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs - config.training.warmup_epochs,
            eta_min=config.training.cosine_eta_min,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.training.warmup_epochs],
        )


    return model, loss_fn, optimizer, device, scheduler


def _build_model(
    config: Config,
    num_classes: int,
    img_size: tuple[int, int, int],
    device: torch.device,
) -> nn.Module:
    """
    Create the specified model and move it to the target device.

    Args:
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
    if config.model_name == "vit":
        model = SimpleViT(
            cfg=config.vit,
            num_classes=num_classes,
            img_size=img_size,
        )
    elif config.model_name == "cnn":
        model = SimpleCNN(
            input_shape=img_size,
            num_classes=num_classes,
            cfg=config.cnn,
        )
    else:
        raise ValueError(f"Unsupported model '{config.model_name}'. "
                         f"Expected one of {'vit', 'cnn'}.")

    return model.to(device)
