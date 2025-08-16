import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer (ViT).

    This module splits the input image into patches, flattens each patch,
    and projects it into a fixed embedding dimension.

    Attributes:
        image_size (tuple): Height and width of the input image.
        patch_size (int): Size of each square patch.
        embedding_dim (int): Output embedding dimension for each patch.
        num_patches (int): Total number of patches (H/P * W/P).
        projection (nn.Linear): Linear layer projecting flattened patches to embedding_dim.
    """
    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        embedding_dim: int,
        in_channels: int
    ):
        super().__init__()

        self.image_height, self.image_width = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        # Validate image size
        if (self.image_height % patch_size != 0
            or self.image_width % patch_size != 0):
            raise ValueError(
                f"Image dimensions ({self.image_height}, {self.image_width}) "
                f"must be divisible by patch size {patch_size}."
            )

        # Number of patches
        self.num_patches_height = self.image_height // patch_size
        self.num_patches_width = self.image_width // patch_size
        self.num_patches = self.num_patches_height * self.num_patches_width

        # Linear projection for flattened patches
        self.projection = nn.Linear(
            in_channels * patch_size * patch_size,
            embedding_dim
        )
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to convert images into patch embeddings.

        Args:
            images (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Patch embeddings of shape (B, N, D)
        """
        B, C, H, W = images.shape

        if H != self.image_height or W != self.image_width:
            raise ValueError(
                f"Input image size ({H}, {W}) does not match expected "
                f"size ({self.image_height}, {self.image_width})"
            )
        if C != self.in_channels:
            raise ValueError(
                f"Input channels ({C}) do not match expected channels "
                f"({self.in_channels})"
            )

        # Extract patches
        patches = images.unfold(2, self.patch_size, self.patch_size) \
                        .unfold(3, self.patch_size, self.patch_size)
        # Shape: (B, C, H/P, W/P, P, P)
        patches = patches.contiguous().view(
            B, C, -1, self.patch_size, self.patch_size
        )  # (B, C, N, P, P)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, N, C, P, P)
        patches = patches.reshape(B, self.num_patches, -1)  # Flatten -> (B, N, C*P*P)

        # Project to embedding dimension
        patch_embeddings = self.projection(patches)  # (B, N, D)

        return patch_embeddings
