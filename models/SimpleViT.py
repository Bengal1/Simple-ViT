from collections.abc import Sequence

import torch
import torch.nn as nn
from typing import Optional
from models.layers.PatchEmbedding import PatchEmbedding
from models.layers.PositionalEncoding import LearnablePositionalEncoding


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
            patch_size: int | tuple[int, int],
            num_classes: int,
            img_size: int | tuple[int, int] | tuple[int, int, int],
            embed_dim: int = 768,
            num_heads: int = 12,
            num_layers: int = 12,
            dim_feedforward: int = 3072,
            dropout: float = 0.1,
            norm_eps: float = 1e-6,
    ):
        """
        Initialize the SimpleViT model.

        Args:
            patch_size (int | tuple[int, int]): Size of each image patch.
            num_classes (int): Number of output classes for classification.
            img_size (tuple[int, int, int]): Image height and width.
            embed_dim (int, optional): Dimensionality of patch embeddings. Default: 768.
            num_heads (int, optional): Number of attention heads. Default: 12.
            num_layers (int, optional): Number of Transformer encoder layers. Default: 12.
            dim_feedforward (int, optional): Hidden size of feedforward layers. Default: 3072.
            dropout (float, optional): Dropout rate. Default: 0.0.
            norm_eps (float, optional): LayerNorm epsilon. Default: 1e-6.
        """
        super().__init__()
        # Set image size as C, H, W format.
        self.img_dim = self._set_input_dimensions(img_size)
        # Validate image patch size relations
        self.n_patches = self._get_number_of_patches(self.img_dim[1:], patch_size)

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(patch_size, self.n_patches,
                                          self.img_dim, embed_dim)

        # --- CLS token (learnable) ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # --- Positional encoding (lazy initialization) ---
        self.pos_encode = LearnablePositionalEncoding(embed_dim)

        # --- Transformer encoder layers ---
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,  # input shape: (B, N, D)
                norm_first=True    # PreNorm like ViT
            )
            for _ in range(num_layers)
        ])

        # --- Layer normalization ---
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # --- Classification head ---
        self.head = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def _set_input_dimensions(input_dim: int | Sequence[int]) -> tuple[int, int, int]:
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
                        is not one, two, or three dimensional.
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
    def _get_number_of_patches(image_size, patch_size):
        # Convert ints to tuples
        H, W = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

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


