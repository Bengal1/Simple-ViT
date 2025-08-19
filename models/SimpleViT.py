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
        img_size: int | tuple[int, int] | None = None,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        """
        Initialize the SimpleViT model.

        Args:
            patch_size (int | tuple[int, int]): Size of each image patch.
            num_classes (int): Number of output classes for classification.
            img_size (int | tuple[int, int], optional): Image height and width.
                                                        If None, PatchEmbedding
                                                        will infer size lazily.
            embed_dim (int, optional): Dimensionality of patch embeddings. Default: 768.
            num_heads (int, optional): Number of attention heads. Default: 12.
            num_layers (int, optional): Number of Transformer encoder layers. Default: 12.
            dim_feedforward (int, optional): Hidden size of feedforward layers. Default: 3072.
            dropout (float, optional): Dropout rate. Default: 0.0.
            norm_eps (float, optional): LayerNorm epsilon. Default: 1e-6.
        """
        super().__init__()

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding(patch_size, embed_dim, img_size)

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


