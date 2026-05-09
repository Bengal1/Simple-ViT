# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
Simple Vision Transformer model for image classification.

This module defines `SimpleViT`, a lightweight Vision Transformer used in
the ViT vs CNN comparison project.

The model is dataset-agnostic and accepts image inputs specified as either
`H`, `(H, W)`, or `(C, H, W)`. Images are divided into fixed-size patches,
projected into an embedding space, enriched with learnable positional
encoding, and processed by stacked Transformer encoder layers.

The network outputs raw logits and is intended for use with
`torch.nn.CrossEntropyLoss`.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from config import ViTConfig
from .layers import LearnablePositionalEncoding, PatchEmbedding


__author__ = "Bengal1"
__all__ = ["SimpleViT"]


# ============================================================
# Simple Vision Transformer
# ============================================================

class SimpleViT(nn.Module):
    """
    Lightweight Vision Transformer for image classification.

    Architecture:
        - Patch embedding layer
        - Learnable CLS token
        - Learnable positional encoding
        - Stack of Transformer encoder layers
        - Layer normalization
        - Linear classification head

    Args:
        cfg (ViTConfig):
            Vision Transformer configuration.
        num_classes (int):
            Number of output classes.
        img_size (int | tuple[int, int] | tuple[int, int, int]):
            Input image size as `H`, `(H, W)`, or `(C, H, W)`.
    """

    def __init__(
        self,
        cfg: ViTConfig,
        num_classes: int,
        img_size: int | tuple[int, int] | tuple[int, int, int],
    ):
        """
        Initialize the SimpleViT model.

        Raises:
            ValueError:
                If model configuration values are invalid or the image size
                is incompatible with the patch size.
        """
        super().__init__()

        self._validate_config(cfg, num_classes)

        self.img_size = self._set_input_dimensions(img_size)
        self.n_patches = self._get_number_of_patches(
            self.img_size,
            cfg.patch_size,
        )

        self.patch_embed = PatchEmbedding(
            patch_size=cfg.patch_size,
            img_size=self.img_size,
            embed_dim=cfg.embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))

        self.pos_encode = LearnablePositionalEncoding(
            embed_dim=cfg.embed_dim,
            num_patches=self.n_patches,
            has_cls_token=True,
        )

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(cfg.num_layers)
        ])

        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.embed_dim, num_classes)

    @staticmethod
    def _validate_config(cfg: ViTConfig, num_classes: int):
        """
        Validate Vision Transformer configuration values.

        Args:
            cfg (ViTConfig):
                Vision Transformer configuration.
            num_classes (int):
                Number of output classes.

        Raises:
            ValueError:
                If any configuration value is invalid.
        """
        if num_classes <= 0:
            raise ValueError(
                f"num_classes must be a positive integer, got {num_classes}."
            )

        if cfg.embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be a positive integer, got {cfg.embed_dim}."
            )

        if cfg.num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {cfg.num_heads}."
            )

        if cfg.num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer, got {cfg.num_layers}."
            )

        if cfg.dim_feedforward <= 0:
            raise ValueError(
                "dim_feedforward must be a positive integer, "
                f"got {cfg.dim_feedforward}."
            )

        if not 0.0 <= cfg.dropout < 1.0:
            raise ValueError(
                f"dropout must be in the range [0, 1), got {cfg.dropout}."
            )

        if cfg.norm_eps <= 0:
            raise ValueError(
                f"norm_eps must be a positive number, got {cfg.norm_eps}."
            )

        if cfg.embed_dim % cfg.num_heads != 0:
            raise ValueError(
                f"embed_dim ({cfg.embed_dim}) must be divisible by "
                f"num_heads ({cfg.num_heads})."
            )

    @staticmethod
    def _set_input_dimensions(
        input_dim: int | Sequence[int],
    ) -> tuple[int, int, int]:
        """
        Normalize input dimensions to `(C, H, W)`.

        Args:
            input_dim (int | Sequence[int]):
                Input size as `H`, `(H, W)`, or `(C, H, W)`.

        Returns:
            tuple[int, int, int]:
                Normalized image size as `(C, H, W)`.

        Raises:
            ValueError:
                If dimensions are non-positive or have unsupported length.
        """
        if isinstance(input_dim, int):
            if input_dim <= 0:
                raise ValueError("Input dimension must be positive.")
            return 1, input_dim, input_dim

        if isinstance(input_dim, (tuple, list)) and len(input_dim) == 2:
            height, width = input_dim
            if height <= 0 or width <= 0:
                raise ValueError("Input dimensions must be positive.")
            return 1, height, width

        if isinstance(input_dim, (tuple, list)) and len(input_dim) == 3:
            channels, height, width = input_dim
            if channels <= 0 or height <= 0 or width <= 0:
                raise ValueError("Input dimensions must be positive.")
            return channels, height, width

        raise ValueError(
            "img_size must be an int, a 2-tuple `(H, W)`, "
            "or a 3-tuple `(C, H, W)`."
        )

    @staticmethod
    def _get_number_of_patches(
        image_size: tuple[int, int, int],
        patch_size: int | tuple[int, int],
    ) -> int:
        """
        Compute the number of image patches.

        Args:
            image_size (tuple[int, int, int]):
                Input image size as `(C, H, W)`.
            patch_size (int | tuple[int, int]):
                Patch size as an integer or `(patch_h, patch_w)`.

        Returns:
            int:
                Number of patches produced from the image.

        Raises:
            ValueError:
                If the patch size is invalid or does not divide the image size.
        """
        _, height, width = image_size

        if isinstance(patch_size, int):
            patch_h, patch_w = patch_size, patch_size
        else:
            patch_h, patch_w = patch_size

        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"patch_size dimensions must be positive, "
                f"got ({patch_h}, {patch_w})."
            )

        if patch_h > height or patch_w > width:
            raise ValueError(
                f"Patch size ({patch_h}x{patch_w}) cannot be larger than "
                f"image size ({height}x{width})."
            )

        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(
                f"Image size ({height}x{width}) must be divisible by "
                f"patch size ({patch_h}x{patch_w})."
            )

        return (height // patch_h) * (width // patch_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            x (torch.Tensor):
                Input batch with shape `(batch_size, C, H, W)`.

        Returns:
            torch.Tensor:
                Class logits with shape `(batch_size, num_classes)`.
        """
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.pos_encode(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)
        x = self.head(x[:, 0])

        return x