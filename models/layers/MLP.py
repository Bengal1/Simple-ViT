import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Feed-forward network (MLP) used inside a Transformer encoder block.

    Args:
        in_features (int): Input feature dimension.
        hidden_features (int): Hidden layer dimension.
        out_features (int): Output feature dimension (usually same as in_features).
        dropout_rate (float): Dropout probability.
    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, dropout_rate: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features


        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
