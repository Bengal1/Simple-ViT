import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for Vision Transformers.

    Splits an input image into non-overlapping patches, flattens each patch,
    and projects it into an embedding space of dimension ``embed_dim``.
    The projection layer can be initialized at construction time or lazily
    on the first forward pass.

    Attributes:
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        embed_dim (int): Embedding dimension of each patch.
        img_height (int | None): Input image height, if known.
        img_width (int | None): Input image width, if known.
        in_channels (int | None): Number of input channels, inferred or set at initialization.
        n_patches (int | None): Total number of patches per image.
        patch_projection (nn.Linear | None): Linear layer projecting flattened patches.
    """

    def __init__(
            self,
            patch_size: int | tuple[int, int],
            img_size: tuple[int, int, int] | None = None,
            embed_dim: int = 768,
    ):
        """
        Initialize the patch embedding layer.

        Args:
            patch_size (int | tuple[int, int]): Patch size as ``int`` for square
                patches or ``(height, width)`` for rectangular patches.
            img_size (tuple[int, int, int] | None): Optional input image size as
                ``(channels, height, width)``. If omitted, the projection layer is
                initialized lazily on the first forward pass.
            embed_dim (int): Output embedding dimension for each patch.

        Raises:
            ValueError: If ``patch_size`` is invalid, ``img_size`` is not in
                ``(C, H, W)`` format, or ``embed_dim`` is not positive.
        """
        super().__init__()

        # --- Patch size ---
        if isinstance(patch_size, int):
            self.patch_height = self.patch_width = patch_size
        elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            self.patch_height, self.patch_width = patch_size
        else:
            raise ValueError(
                f"Patch size must be an int or a tuple (H, W), "
                f"but got {patch_size!r}"
            )

        if embed_dim <= 0:
            raise ValueError(
                f"embed_dim must be positive number, "
                f"but got {embed_dim!r}."
            )
        self.embed_dim = embed_dim

        # --- Image size & projection ---
        self.img_height: int | None = None
        self.img_width: int | None = None
        self.in_channels: int | None = None
        self.n_patches: int | None = None
        self.patch_projection: nn.Linear | None = None

        if img_size is not None:
            if not (isinstance(img_size, (tuple, list)) and len(img_size) == 3):
                raise ValueError(
                    f"Image size must be a tuple (C, H, W), but got {img_size!r}"
                )
            self.in_channels, self.img_height, self.img_width = img_size
            
            # Compute number of patches (floor division ensures tiling)
            self.n_patches = (self.img_height // self.patch_height) * \
                             (self.img_width // self.patch_width)

            # Initialize projection immediately
            patch_dim = self.in_channels * self.patch_height * self.patch_width
            self.patch_projection = nn.Linear(patch_dim, self.embed_dim)

    @staticmethod
    def _get_input_size(
            x: torch.Tensor
    ) -> tuple[torch.Tensor, int, int, int, int]:
        """
        Convert input to 4D shape (B, C, H, W) if needed and return both the
        converted tensor and its dimensions.

        Supported inputs:
            - 2D: (H, W)         -> (1, 1, H, W)
            - 3D: (C, H, W)      -> (1, C, H, W)
            - 4D: (B, C, H, W)   -> unchanged

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, int, int, int, int]:
                The reshaped tensor and its dimensions as (x, B, C, H, W).

        Raises:
            ValueError: If the input tensor is not 2D, 3D, or 4D.
        """
        if x.ndim == 4:
            B, C, H, W = x.shape
        elif x.ndim == 3:
            x = x.unsqueeze(0)
            B, C, H, W = x.shape
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            B, C, H, W = x.shape
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D input, got {x.ndim}D")

        return x, B, C, H, W

    def _initialize_projection(
            self,
            C: int,
            H: int,
            W: int,
            device: torch.device
    ) -> None:
        """
        Lazily initialize the projection layer and related image metadata.

        This method sets the expected input dimensions, validates consistency, and
        creates the linear projection used to embed flattened patches.

        Args:
            C (int): Number of input channels.
            H (int): Input image height.
            W (int): Input image width.
            device (torch.device): Device on which to create the projection layer.

        Raises:
            ValueError: If the input dimensions are inconsistent with the stored
                image structure or are incompatible with the patch size.
        """
        # safety check: if already initialized, exit silently
        if self.patch_projection is not None:
            return

        # --- Set image dimensions if not already fixed ---
        if self.img_height is None:
            self.img_height = H
        elif self.img_height != H:
            raise ValueError(
                f"Image height mismatch: expected {self.img_height}, got {H}"
            )

        if self.img_width is None:
            self.img_width = W
        elif self.img_width != W:
            raise ValueError(
                f"Image width mismatch: expected {self.img_width}, but got {W}"
            )

        if self.in_channels is None:
            self.in_channels = C
        elif self.in_channels != C:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, got {C}"
            )

        # --- Compute and validate number of patches ---
        if H % self.patch_height != 0 or W % self.patch_width != 0:
            raise ValueError(
                f"Input image size ({H}x{W}) must be divisible by patch size "
                f"({self.patch_height}x{self.patch_width})."
            )

        num_patches = (self.img_height // self.patch_height) * \
                      (self.img_width // self.patch_width)

        if self.n_patches is None:
            self.n_patches = num_patches
        elif self.n_patches != num_patches:
            raise ValueError(
                f"Number of patches mismatch: expected {self.n_patches}, "
                f"but got {num_patches}"
            )

        # --- Initialize projection layer ---
        patch_dim = self.in_channels * self.patch_height * self.patch_width
        self.patch_projection = nn.Linear(patch_dim, self.embed_dim).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert an input image batch into a sequence of patch embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape ``(H, W)``, ``(C, H, W)``, or
                ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Patch embeddings of shape
                ``(B, num_patches, embed_dim)``.

        Raises:
            ValueError: If the input image size does not match the initialized
                image dimensions.
        """
        # --- Ensure input is 4D ---
        x, B, C, H, W = self._get_input_size(x)

        # --- If not initialized --> Lazy initialization ---
        if self.patch_projection is None:
            self._initialize_projection(C, H, W, x.device)

        # --- Check input image size ---
        if H != self.img_height or W != self.img_width:
            raise ValueError(
                f"Input image size ({H}x{W}) doesn't match the initialized size "
                f"({self.img_height}x{self.img_width})"
            )

        # --- Patches extraction ---
        patches = x.unfold(2, self.patch_height, self.patch_height) \
                   .unfold(3, self.patch_width, self.patch_width)
        # Combine patch grid into a single dimension
        patches = patches.contiguous().view(B, C, -1, self.patch_height, self.patch_width)
        # Move patches to sequence dimension (B, n_patches, C, ph, pw)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        # Flatten each patch to vector (B, n_patches, C*ph*pw)
        patches = patches.view(B, self.n_patches, -1)

        # --- Linear projection ---
        embedded_patches = self.patch_projection(patches)

        return embedded_patches


