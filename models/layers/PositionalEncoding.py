import torch
import torch.nn as nn
from torch import Tensor


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for Vision Transformers (ViT).

    Maintains a trainable positional embedding for each patch
    (and optional CLS token) and adds it to the input patch embeddings.

    Attributes:
        embed_dim (int): Dimensionality of patch embeddings.
        num_patches (int | None): Number of image patches (can be inferred lazily).
        has_cls_token (bool): Whether to include an extra position for a CLS token.
        pos_embedding (nn.Parameter | None): Learnable positional embeddings.
    """

    def __init__(
            self,
            embed_dim: int,
            num_patches: int | None = None,
            has_cls_token: bool = True):
        """
        Initialize LearnablePositionalEncoding.

        Args:
            num_patches (int | None): Number of image patches. If None, will be
                                       inferred on first forward pass.
            embed_dim (int): Embedding dimension of each patch.
            has_cls_token (bool): If True, includes an extra position for a CLS token.
        """
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.has_cls_token = has_cls_token

        # Placeholder for lazy initialization
        self.pos_embedding: nn.Parameter | None = None

    def _initialize_pos_embedding(self, seq_len: int) -> None:
        """
        Lazy initialization of positional embeddings based on actual sequence length.

        Args:
            seq_len (int): Number of tokens (patches + optional CLS) in input.
        """
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, self.embed_dim))
        # Initialize using truncated normal (ViT default)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to input embeddings.

        Args:
            x (Tensor): Input embeddings of shape (B, N, D)
                        B = batch size
                        N = number of tokens (patches + optional CLS)
                        D = embedding dimension.

        Returns:
            Tensor: Positionally encoded embeddings of shape (B, N, D).

        Raises:
            ValueError: If input tensor is not 3-dimensional.
            ValueError: If input sequence length does not match positional embeddings.
        """
        # --- Check input dimension ---
        if x.ndim != 3:
            raise ValueError(
                f"Expected input tensor with 3 dimensions (B, N, D), got {x.ndim}D"
            )

        patches_num = x.size(1)

        # --- Lazy initialization ---
        if self.pos_embedding is None:
            self._initialize_pos_embedding(patches_num)

        # Ensure number of patches matches positional embeddings
        if patches_num != self.pos_embedding.size(1):
            raise ValueError(
                f"Expected sequence length {self.pos_embedding.size(1)}, got {patches_num}"
            )

        # --- Add positional embeddings ---
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
