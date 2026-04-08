# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
===========================
         SimpleViT
===========================

A lightweight Vision Transformer for image classification.

Architecture:
    - Patch embedding layer (image → sequence of patches)
    - Learnable positional encoding
    - Stacked Transformer encoder blocks (Multi-head Self-Attention + MLP)
    - Classification head (CLS token or pooled representation)

The model is dataset-agnostic and accepts inputs of shape (C, H, W).
Images are split into fixed-size patches, projected into an embedding
space, and processed using self-attention.

Outputs raw logits and is intended for use with
`torch.nn.CrossEntropyLoss`.
"""
__author__="Bengal1"


import torch
import torch.nn as nn
from collections.abc import Sequence

from .layers import PatchEmbedding, LearnablePositionalEncoding
from ..config import ViTConfig


class SimpleViT(nn.Module):
    """
    Simple Vision Transformer (ViT) for image classification.

    Processes images into patch embeddings, adds learnable positional encodings,
    passes through Transformer encoder layers, and outputs class predictions.

    Attributes:
        patch_embed (PatchEmbedding): Converts images into patch embeddings.
        cls_token (nn.Parameter): Learnable CLS token for classification.
        pos_encode (LearnablePositionalEncoding): Adds positional embeddings to patches.
        encoder_layers (nn.ModuleList): Stack of TransformerEncoderLayer modules.
        norm (nn.LayerNorm): Normalizes the output of the Transformer.
        head (nn.Linear): Projects CLS token output to class logits.
    """

    def __init__(
            self,
            cfg: ViTConfig,
            num_classes: int,
            img_size: int | tuple[int, int] | tuple[int, int, int],

    ):
        """
        Initialize the SimpleViT model.

        Args:
            cfg (ViTConfig): Configuration object containing model hyperparameters.
            num_classes (int): Number of output classes.
            img_size (int | tuple[int, int] | tuple[int, int, int]):
                                                Input image size as H, W or C, H, W.
        """
        super().__init__()
        # --- Validate model configuration ---
        if num_classes <= 0:
            raise ValueError(
                f"num_classes must be a positive integer, but got {num_classes}"
            )

        if cfg.embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be a positive integer, but got {cfg.embed_dim}"
            )

        if cfg.num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, but got {cfg.num_heads}"
            )

        if cfg.num_layers <= 0:
            raise ValueError(
                f"num_layers must be a positive integer, but got {cfg.num_layers}"
            )

        if cfg.dim_feedforward <= 0:
            raise ValueError(
                f"dim_feedforward must be a positive integer, but got {cfg.dim_feedforward}"
            )

        if not 0.0 <= cfg.dropout < 1.0:
            raise ValueError(
                f"dropout must be in the range [0, 1), but got {cfg.dropout}"
            )

        if cfg.norm_eps <= 0:
            raise ValueError(
                f"norm_eps must be a positive number, but got {cfg.norm_eps}"
            )

        if cfg.embed_dim % cfg.num_heads != 0:
            raise ValueError(
                f"embed_dim ({cfg.embed_dim}) must be divisible by num_heads ({cfg.num_heads})"
            )

        # Set image size as C, H, W format.
        self.img_size = self._set_input_dimensions(img_size)
        # Validate image patch size relations
        self.n_patches = self._get_number_of_patches(self.img_size, cfg.patch_size)

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(
            cfg.patch_size,
            self.img_size,
            cfg.embed_dim
        )

        # --- CLS token (learnable) ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))

        # --- Positional encoding ---
        self.pos_encode = LearnablePositionalEncoding(
            cfg.embed_dim,
            num_patches=self.n_patches,
            has_cls_token=True
        )

        # --- Transformer encoder layers ---
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,  # input shape: (B, N, D)
                norm_first=True
            )
            for _ in range(cfg.num_layers)
        ])

        # --- Layer normalization ---
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.norm_eps)

        # --- Classification head ---
        self.head = nn.Linear(cfg.embed_dim, num_classes)


    @staticmethod
    def _set_input_dimensions(
            input_dim: int | Sequence[int]
    ) -> tuple[int, int, int]:
        """
        Normalize the input dimensions into a 3-tuple of positive integers.

        This method accepts either:
          - a single integer (treated as square dimensions: (1, dim, dim)),
          - a sequence of two integers (treated as (1, height, width)),
          - a sequence of three integers (treated as (channels, height, width)).

        Args:
            input_dim (int | Sequence[int]): Input dimension specification.

        Returns:
            tuple[int, int, int]: A 3-tuple representing (channels, height, width).

        Raises:
            ValueError: If any dimension is non-positive, or if the input
                        is not one, two, or three-dimensional.
        """
        if isinstance(input_dim, int):
            if input_dim <= 0:
                raise ValueError("dimension must be a positive integer")
            return 1, input_dim, input_dim

        elif isinstance(input_dim, (tuple, list)) and len(input_dim) == 2:
            if input_dim[0] <= 0 or input_dim[1] <= 0:
                raise ValueError("all dimensions must be positive")
            return 1, input_dim[0], input_dim[1]

        elif isinstance(input_dim, (tuple, list)) and len(input_dim) == 3:
            if input_dim[0] <= 0 or input_dim[1] <= 0 or input_dim[2] <= 0:
                raise ValueError("all dimensions must be positive")
            return input_dim[0], input_dim[1], input_dim[2]

        else:
            raise ValueError("input size must be one, two, or three dimensional")


    @staticmethod
    def _get_number_of_patches(
            image_size: tuple[int, int, int],
            patch_size: int | tuple[int, int]
    ) -> int:

        H, W =  image_size[1:]
        patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        # Check patch size positive
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"patch_size dimensions must be positive, but got ({patch_h}, {patch_w})"
            )
        # Check patch smaller than image
        if patch_h > H or patch_w > W:
            raise ValueError(
                f"Patch size ({patch_h}x{patch_w}) cannot be larger than image size ({H}x{W}).")
        # Check divisibility
        if H % patch_h != 0 or W % patch_w != 0:
            raise ValueError(
                f"Image size ({H}x{W}) must be divisible by patch size ({patch_h}x{patch_w}).")


        # Compute number of patches
        n_patches = (H // patch_h) * (W // patch_w)

        return n_patches



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vision Transformer.

        Args:
            x (Tensor): Input images, shape (B, C, H, W)

        Returns:
            Tensor: Class logits, shape (B, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, D)

        # CLS token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1) # (B, n_patches + 1, D)

        # Positional encoding
        x = self.pos_encode(x)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Layer norm
        x = self.norm(x)

        # Classification head on CLS token
        return self.head(x[:, 0])


