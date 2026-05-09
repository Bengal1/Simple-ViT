# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
Positional encoding layers for Vision Transformer models.

This module defines `LearnablePositionalEncoding`, which adds trainable
positional embeddings to patch-token sequences. It supports an optional CLS
token and can initialize positional embeddings either eagerly or lazily.
"""

import torch
import torch.nn as nn


__author__ = "Bengal1"
__all__ = ["LearnablePositionalEncoding"]


# ============================================================
# Learnable Positional Encoding
# ============================================================

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for Vision Transformers.

    The layer maintains a trainable positional embedding for each token in
    the sequence and adds it to the input embeddings.

    Args:
        embed_dim (int):
            Embedding dimension of each token.
        num_patches (int | None, optional):
            Number of image patches. If provided, positional embeddings are
            initialized during construction. If None, they are initialized
            lazily on the first forward pass.
        has_cls_token (bool, optional):
            If True, reserves one additional position for a CLS token.
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches: int | None = None,
        has_cls_token: bool = True,
    ):
        """
        Initialize the learnable positional encoding layer.

        Raises:
            ValueError:
                If `embed_dim` or `num_patches` is invalid.
        """
        super().__init__()

        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be a positive integer, got {embed_dim!r}."
            )

        if num_patches is not None and num_patches <= 0:
            raise ValueError(
                f"num_patches must be a positive integer, got {num_patches!r}."
            )

        self.embed_dim = embed_dim
        self.has_cls_token = has_cls_token

        self.num_tokens: int | None = None
        self.pos_embedding: nn.Parameter | None = None

        if num_patches is not None:
            self.num_tokens = num_patches + 1 if has_cls_token else num_patches
            self.pos_embedding = nn.Parameter(
                torch.empty(1, self.num_tokens, self.embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _initialize_pos_embedding(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> None:
        """
        Lazily initialize positional embeddings.

        Args:
            num_tokens (int):
                Sequence length, including CLS token if present.
            device (torch.device):
                Device on which to create the positional embeddings.

        Raises:
            ValueError:
                If the sequence length does not match the expected length.
        """
        if self.pos_embedding is not None:
            return

        if num_tokens <= 0:
            raise ValueError(
                f"num_tokens must be positive, got {num_tokens}."
            )

        if self.num_tokens is None:
            self.num_tokens = num_tokens
        elif self.num_tokens != num_tokens:
            raise ValueError(
                f"Expected sequence length {self.num_tokens}, got {num_tokens}."
            )

        self.pos_embedding = nn.Parameter(
            torch.empty(1, self.num_tokens, self.embed_dim, device=device)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to input token embeddings.

        Args:
            x (torch.Tensor):
                Input tensor with shape `(batch_size, num_tokens, embed_dim)`.

        Returns:
            torch.Tensor:
                Positionally encoded tensor with the same shape as input.

        Raises:
            ValueError:
                If the input tensor shape is invalid or incompatible with the
                initialized positional embeddings.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input tensor with shape `(B, N, D)`, got {x.ndim}D."
            )

        if x.size(2) != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, got {x.size(2)}."
            )

        num_tokens = x.size(1)

        if self.pos_embedding is None:
            self._initialize_pos_embedding(num_tokens, x.device)

        if num_tokens != self.num_tokens:
            raise ValueError(
                f"Expected sequence length {self.num_tokens}, got {num_tokens}."
            )

        return x + self.pos_embedding

