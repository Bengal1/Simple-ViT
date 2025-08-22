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

    def __init__(self,
                 patch_size: int | tuple[int, int],
                 embed_dim: int = 768,
                 img_size: int | tuple[int, int] | None = None):
        """
        Initialize PatchEmbedding.

        Args:
            patch_size (int | tuple[int, int]): Size of each patch (height, width).
                                                If int, patch is square.
            embed_dim (int): Output embedding dimension of each patch.
            img_size (int | tuple[int, int] | None): Optional image size for pre-initialization.
                                                     If None, will infer from input on first forward.

        Raises:
            ValueError: If `patch_size` or `img_size` does not have 1 or 2 dimensions.
        """
        super().__init__()

        # --- Patch size ---
        if isinstance(patch_size, int):
            self.patch_height = self.patch_width = patch_size
        elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            self.patch_height, self.patch_width = patch_size
        else:
            raise ValueError("Patch size must be one or two dimensional")

        self.embed_dim = embed_dim

        # --- Image size ---
        if img_size is not None:
            if isinstance(img_size, int):
                self.img_height = self.img_width = img_size
            elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
                self.img_height, self.img_width = img_size
            else:
                raise ValueError("Image size must be one or two dimensional")
        else:
            self.img_height = None
            self.img_width = None

        # --- Linear projection placeholder ---
        self.n_patches = None
        self.patch_projection = None


    @staticmethod
    def get_input_size(x: torch.Tensor) -> tuple[int, int, int, int]:
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


    def _initialize_projection(self, C: int, H: int, W: int) -> None:
        """
        Lazy initialization of the patch projection layer and related attributes.
        Only runs if the projection layer has not been created yet.

        Args:
            C (int): Number of input channels.
            H (int): Height of the input image.
            W (int): Width of the input image.
        """
        if self.patch_projection is None:
            # Set image dimensions if not previously set
            if self.img_height is None:
                self.img_height = H
            if self.img_width is None:
                self.img_width = W

            # Compute number of patches
            self.n_patches = (self.img_height // self.patch_height) * \
                             (self.img_width // self.patch_width)

            # Initialize linear projection using actual channels
            patch_dim = C * self.patch_height * self.patch_width
            self.patch_projection = nn.Linear(patch_dim, self.embed_dim)


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
        B, C, H, W = self.get_input_size(x)

        # --- Lazy initialize projection ---
        self._initialize_projection(C, H, W)

        # --- Check input image size ---
        if H != self.img_height or W != self.img_width:
            raise ValueError(
                f"Input image size ({H}x{W}) doesn't match the initialized size "
                f"({self.img_height}x{self.img_width})"
            )

        # --- Extract patches ---
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


    def get_patches_num(self) -> int:
        """
        Return the number of patches per image.

        Returns:
            int: Total number of patches (n_patches).

        Raises:
            RuntimeError: If the number of patches is not yet initialized.
        """
        if self.n_patches is not None:
            return self.n_patches

        # If image size is known but projection not yet initialized, compute n_patches
        if self.img_height is not None and self.img_width is not None:
            return (self.img_height // self.patch_height) * (
                        self.img_width // self.patch_width)

        raise RuntimeError(
            "Number of patches not initialized. "
            "Pass an input through the layer or provide img_size at initialization."
        )

