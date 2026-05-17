# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
Training setup utilities.

This module builds the core training components:
    - model
    - loss function
    - optimizer
    - device
    - optional learning-rate scheduler

It acts as the bridge between the global configuration object and the
training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from models import SimpleCNN, SimpleViT
from utils import get_device


__author__ = "Bengal1"
__all__ = ["setup_model_for_training"]


# ============================================================
# Training Setup
# ============================================================

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
    Initialize model, loss, optimizer, device, and scheduler.

    Args:
        config (Config):
            Global configuration object.
        num_classes (int):
            Number of output classes.
        img_size (tuple[int, int, int]):
            Input image shape as `(C, H, W)`.

    Returns:
        tuple:
            model, loss function, optimizer, device, and optional scheduler.
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

    scheduler = _build_scheduler(
        config=config,
        optimizer=optimizer,
    )

    return model, loss_fn, optimizer, device, scheduler


def _build_model(
    config: Config,
    num_classes: int,
    img_size: tuple[int, int, int],
    device: torch.device,
) -> nn.Module:
    """
    Build the selected model and move it to the target device.

    Args:
        config (Config):
            Global configuration object.
        num_classes (int):
            Number of output classes.
        img_size (tuple[int, int, int]):
            Input image shape as `(C, H, W)`.
        device (torch.device):
            Target computation device.

    Returns:
        nn.Module:
            Initialized model on the selected device.

    Raises:
        ValueError:
            If the configured model name is unsupported.
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
        raise ValueError(
            f"Unsupported model '{config.model_name}'. "
            "Expected one of: 'vit', 'cnn'."
        )

    return model.to(device)


def _build_scheduler(
    config: Config,
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler.LRScheduler | None:
    """
    Build the learning-rate scheduler.

    Uses linear warmup followed by cosine annealing when enabled.

    Args:
        config (Config):
            Global configuration object.
        optimizer (optim.Optimizer):
            Optimizer whose learning rate will be scheduled.

    Returns:
        optim.lr_scheduler.LRScheduler | None:
            Scheduler if enabled, otherwise None.

    Raises:
        ValueError:
            If scheduler configuration is invalid.
    """
    if not config.training.use_scheduler:
        return None

    if config.training.warmup_epochs >= config.training.epochs:
        raise ValueError(
            "warmup_epochs must be smaller than total training epochs."
        )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config.training.warmup_start_factor,
        total_iters=config.training.warmup_epochs,
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs - config.training.warmup_epochs,
        eta_min=config.training.cosine_eta_min,
    )

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.training.warmup_epochs],
    )