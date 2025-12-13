import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Minimal Patch Embedding layer for Vision Transformers.

    Splits an input image into patches and projects each patch into a
    vector of size `embed_dim` using a linear layer.

    Attributes:
        patch_height (int): Height of each patch.
        patch_width (int): Width of each patch.
        embed_dim (int): Output embedding dimension for each patch.
        img_height (int | None): Height of input image (if known at init).
        img_width (int | None): Width of input image (if known at init).
        n_patches (int | None): Total number of patches per image.
        patch_projection (nn.Linear | None): Linear layer to project flattened patches.
    """

    def __init__(
            self,
            patch_size: int | tuple[int, int],
            img_size: tuple[int, int, int] | None = None,
            embed_dim: int = 768,
    ):
        """
        Initialize PatchEmbedding.

        Args:
            patch_size (int | tuple[int, int]): Size of each patch (height, width).
                If int, the patch is square.
            img_size (tuple[int, int, int] | None): Optional image size (C, H, W).
                If None, the projection layer will be lazily initialized on the
                first forward pass.
            embed_dim (int): Output embedding dimension of each patch. Default: 768.

        Raises:
            ValueError: If `patch_size` is invalid or `img_size` does not follow (C, H, W).
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
                f"but got size of {patch_size!r}"
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
        self.n_patches: int | None = None
        self.patch_projection: nn.Linear | None = None

        if img_size is not None:
            if not (isinstance(img_size, (tuple, list)) and len(img_size) == 3):
                raise ValueError(
                    f"Image size must be a tuple (C, H, W), but got {img_size!r}"
                )
            C, self.img_height, self.img_width = img_size
            
            # Compute number of patches (floor division ensures tiling)
            self.n_patches = (self.img_height // self.patch_height) * \
                             (self.img_width // self.patch_width)

            # Initialize projection immediately
            patch_dim = C * self.patch_height * self.patch_width
            self.patch_projection = nn.Linear(patch_dim, self.embed_dim)


    @staticmethod
    def _get_input_size(x: torch.Tensor) -> tuple[int, int, int, int]:
        """
        Ensure input is 4D: (B, C, H, W). Converts 2D or 3D inputs into 4D by adding
        batch and/or channel dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape 2D, 3D, or 4D.

        Returns:
            tuple[int, int, int, int]: 4D tensor shape (B, C, H, W).

        Raises:
            ValueError: If input tensor has dimensions other than 2, 3, or 4.
        """
        if x.ndim == 4:
            B, C, H, W = x.shape
        elif x.ndim == 3:
            C, H, W = x.shape
            B = 1
        elif x.ndim == 2:
            H, W = x.shape
            B, C = 1, 1
        else:
            raise ValueError(f"Expected 2D, 3D, or 4D input, got {x.ndim}D")
        return B, C, H, W

    def _initialize_projection(self,
                               C: int, H: int, W: int,
                               device: torch.device
    ) -> None:
        """
        Lazily initialize the patch projection layer and related attributes.
        This should only be called once when the first input is seen.

        Args:
            C (int): Number of input channels.
            H (int): Height of the input image.
            W (int): Width of the input image.
            device (torch.device): Target device for lazy initialization.

        Raises:
            ValueError: If a mismatch is detected in image size or patch count.
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

        # --- Compute and validate number of patches ---
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
        patch_dim = C * self.patch_height * self.patch_width
        self.patch_projection = nn.Linear(patch_dim, self.embed_dim).to(device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: split image into patches, flatten, and project each patch.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or convertible
                              2D/3D tensor.

        Returns:
            torch.Tensor: Tensor of shape (B, n_patches, embed_dim) containing
                          the patch embeddings.

        Raises:
            ValueError: If the input image size does not match the initialized size.
        """
        # --- Ensure input is 4D ---
        B, C, H, W = self._get_input_size(x)

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


