import torch
import torch.nn as nn
from torch import Tensor


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for Vision Transformers (ViT).

    This module maintains a trainable positional embedding for each patch
    (and an optional CLS token) and adds it to the input patch embeddings.

    Args:
        num_patches (int): Number of image patches.
        embed_dim (int): Dimensionality of patch embeddings.
        has_cls_token (bool, optional): If True, includes an extra position
            for a CLS token. Default: True.
    """

    def __init__(self, num_patches: int, embed_dim: int, has_cls_token: bool = True):
        super().__init__()

        total_tokens = num_patches + (1 if has_cls_token else 0)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.empty(1, total_tokens, embed_dim)
        )

        # Initialization: truncated normal, std=0.02 (ViT default)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input embeddings of shape (B, N, D)
                B = batch size
                N = number of tokens (patches + optional CLS)
                D = embedding dimension.

        Returns:
            Tensor: Positionally encoded embeddings of shape (B, N, D).
        """
        if x.size(1) != self.pos_embedding.size(1):
            raise ValueError(
                f"Expected sequence length {self.pos_embedding.size(1)}, "
                f"but got {x.size(1)}"
            )

        # Add positional embeddings
        return x + self.pos_embedding.to(x.device)



# class PositionalEncoding2D(nn.Module):
#     """
#     2D Sinusoidal Positional Encoding for Vision Transformer (ViT).
#
#     Each patch embedding receives a D-dimensional vector:
#         - First half encodes width (columns)
#         - Second half encodes height (rows)
#     """
#     def __init__(
#         self,
#         embedding_dim: int,
#         patch_grid_size: tuple[int, int]
#     ):
#         """
#         Args:
#             embedding_dim (int): Dimension of each patch embedding (D)
#             patch_grid_size (tuple[int, int]): Number of patches along (height, width) -> (H, W)
#         """
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         self.num_patches_height, self.num_patches_width = patch_grid_size
#
#         if embedding_dim % 2 != 0:
#             raise ValueError(
#                 "Embedding dimension must be even for 2D positional encoding"
#             )
#
#         # Precompute the positional encoding tensor and register as buffer
#         self.register_buffer(
#             'positional_encoding',
#             self._build_2d_positional_encoding()
#         )
#
#     def _build_1d_sinusoidal_encoding(self, length: int) -> torch.Tensor:
#         """
#         Build 1D sinusoidal positional encoding for a single dimension.
#
#         Args:
#             length (int): Number of positions (H or W)
#
#         Returns:
#             torch.Tensor: Shape (length, embedding_dim // 2)
#         """
#         dim_half = self.embedding_dim // 2
#         positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # (length, 1)
#         div_term = torch.exp(
#             torch.arange(0, dim_half, 2, dtype=torch.float32) *
#             -(torch.log(torch.tensor(10000.0)) / dim_half)
#         )
#
#         encoding = torch.zeros(length, dim_half)
#         encoding[:, 0::2] = torch.sin(positions * div_term)
#         encoding[:, 1::2] = torch.cos(positions * div_term)
#
#         return encoding  # (length, dim_half)
#
#     def _build_2d_positional_encoding(self) -> torch.Tensor:
#         """
#         Build the full 2D positional encoding for all patches.
#
#         Returns:
#             torch.Tensor: Shape (1, num_patches_height*num_patches_width, embedding_dim)
#         """
#         # 1D encodings
#         width_encoding = self._build_1d_sinusoidal_encoding(self.num_patches_width)   # (W, D/2)
#         height_encoding = self._build_1d_sinusoidal_encoding(self.num_patches_height) # (H, D/2)
#
#         # Combine into 2D grid
#         grid_encoding = torch.zeros(
#             self.num_patches_height,
#             self.num_patches_width,
#             self.embedding_dim
#         )
#         grid_encoding[:, :, :self.embedding_dim // 2] = width_encoding.unsqueeze(0)    # broadcast along height
#         grid_encoding[:, :, self.embedding_dim // 2:] = height_encoding.unsqueeze(1)   # broadcast along width
#
#         # Flatten grid to sequence: (N, D)
#         grid_encoding = grid_encoding.view(
#             self.num_patches_height * self.num_patches_width,
#             self.embedding_dim
#         )
#
#         # Add batch dimension: (1, N, D)
#         return grid_encoding.unsqueeze(0)
#
#     def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
#         """
#         Add 2D positional encoding to patch embeddings.
#
#         Args:
#             patch_embeddings (torch.Tensor): Shape (batch_size, num_patches, embedding_dim)
#
#         Returns:
#             torch.Tensor: Shape (batch_size, num_patches, embedding_dim)
#         """
#         return patch_embeddings + self.positional_encoding[:, :patch_embeddings.size(1), :]
