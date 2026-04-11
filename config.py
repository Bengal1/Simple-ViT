# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
Central configuration module.

Defines structured dataclass-based configurations for model architecture,
training, and optimization. The default instance (`config`) acts as the
runtime source of truth and can be overridden programmatically or via CLI.

Components:
    - ViTConfig: Vision Transformer hyperparameters
    - CNNConfig: Convolutional network hyperparameters
    - TrainingConfig: Training process settings
    - OptimConfig: Optimizer parameters
    - Config: Aggregated configuration container
"""
__author__ = "Bengal1"

from dataclasses import dataclass, field


# ======================================================================
# Model Configurations
# ======================================================================

@dataclass
class ViTConfig:
    """Vision Transformer (ViT) hyperparameters."""
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    patch_size: int | tuple[int, int] = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    norm_eps: float = 1e-6


@dataclass
class CNNConfig:
    """Simple CNN hyperparameters."""
    conv1_out_channels: int = 32
    conv2_out_channels: int = 64
    conv_kernel_size: int = 3
    pool_kernel_size: int = 2
    pool_stride: int = 2

    fc2_in: int = 512

    dropout1_rate: float = 0.35
    dropout2_rate: float = 0.25


# ======================================================================
# Training Configuration
# ======================================================================

@dataclass
class TrainingConfig:
    """Training process configuration."""
    batch_size: int = 128
    epochs: int = 100
    validation_split: float = 0.2

    # --- Optimization behavior ---
    accumulation_steps: int = 7
    max_grad_clip: float | None = 1.0

    # --- Regularization ---
    label_smooth: float = 0.1

    # --- Early stopping ---
    patience: int = 5


# ======================================================================
# Optimizer Configuration
# ======================================================================

@dataclass
class OptimConfig:
    """Optimizer configuration."""
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9


# ======================================================================
# Global Configuration
# ======================================================================

@dataclass
class Config:
    """
    Global configuration container.

    Holds runtime selections (model, dataset) and aggregates all
    sub-configurations for model, training, and optimization.
    """

    # --- Runtime selection ---
    model_name: str = "vit"   # {"vit", "cnn"}
    dataset: str = "mnist"    # {"mnist", "cifar10", "tiny_imagenet"}

    # --- Model configs ---
    vit: ViTConfig = field(default_factory=ViTConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)

    # --- Training & optimization ---
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def update_from_args(self, args) -> "Config":
        """
        Update runtime configuration fields from CLI arguments.

        This method applies user-provided command-line arguments to the
        configuration instance and validates that the selected options
        are supported.

        Args:
            args:
                Parsed argparse namespace containing at least:
                    - dataset (str)
                    - model (str)

        Returns:
            Config:
                Updated configuration instance.

        Raises:
            ValueError:
                If `model` or `dataset` is not supported.
        """
        self.dataset = args.dataset
        self.model_name = args.model

        if self.model_name not in {"vit", "cnn"}:
            raise ValueError(f"Invalid model_name: {self.model_name}")

        if self.dataset not in {"mnist", "cifar10", "tiny_imagenet"}:
            raise ValueError(f"Invalid dataset: {self.dataset}")

        return self



# ======================================================================
# Default Configuration Instance
# ======================================================================

config = Config()