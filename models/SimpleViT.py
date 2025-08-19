import torch
import torch.nn as nn
from models.layers.PatchEmbedding import PatchEmbedding
from models.layers.PositionalEncoding import LearnablePositionalEncoding


class SimpleViT(nn.Module):

    def __init__(self,
                 num_classes: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 norm_eps: float = 1e-6):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding()

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional Encoding
        self.pos_encode = LearnablePositionalEncoding()

        # Transformer Encoder Layers
        self.encoder_layers = self.encoder_layers = torch.nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,  # (B, N, D)
                norm_first=True  # PreNorm like ViT
            )
            for _ in range(num_layers)
        ])

        # Normalizing Layer
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # Classification Projection
        self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x):

        x = self.patch_embed(x)

        cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)

        x = torch.cat([cls, x], dim=1)

        x = self.pos_encode(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        cls_token_output = x[:, 0]

        return self.head(cls_token_output)

