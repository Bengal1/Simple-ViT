# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
Simple CNN model for image classification.

This module defines `SimpleCNN`, a compact convolutional neural network used
as the CNN baseline in the ViT vs CNN comparison project.

The model is dataset-agnostic and accepts input images with shape `(C, H, W)`.
The classifier input size is computed dynamically during initialization, making
the model compatible with datasets such as MNIST, CIFAR-10, and Tiny ImageNet.

The network outputs raw logits and is intended for use with
`torch.nn.CrossEntropyLoss`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CNNConfig


__author__ = "Bengal1"
__all__ = ["SimpleCNN"]


# ============================================================
# Simple CNN
# ============================================================

class SimpleCNN(nn.Module):
    """
    Compact convolutional neural network for image classification.

    Architecture:
        - Two convolutional blocks:
            Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Dropout
        - One hidden fully connected layer
        - One output layer producing class logits

    Args:
        input_shape (tuple[int, int, int]):
            Input image shape as `(C, H, W)`.
        num_classes (int):
            Number of output classes.
        cfg (CNNConfig):
            CNN architecture configuration.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int = 10,
        cfg: CNNConfig = CNNConfig(),
    ):
        """
        Initialize the SimpleCNN model.

        Raises:
            ValueError:
                If the input spatial dimensions are too small for the
                configured convolution and pooling layers.
        """
        super().__init__()

        input_channels, height, width = input_shape

        for _ in range(2):
            height = height - cfg.conv_kernel_size + 1
            width = width - cfg.conv_kernel_size + 1

            height = (height - cfg.pool_kernel_size) // cfg.pool_stride + 1
            width = (width - cfg.pool_kernel_size) // cfg.pool_stride + 1

        if height <= 0 or width <= 0:
            raise ValueError(
                "Input spatial dimensions are too small for the configured "
                "CNN architecture."
            )

        fc1_in = cfg.conv2_out_channels * height * width

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=cfg.conv1_out_channels,
            kernel_size=cfg.conv_kernel_size,
        )
        self.batch1 = nn.BatchNorm2d(num_features=cfg.conv1_out_channels)
        self.max1 = nn.MaxPool2d(
            kernel_size=cfg.pool_kernel_size,
            stride=cfg.pool_stride,
        )
        self.dropout1 = nn.Dropout(p=cfg.dropout1_rate)

        self.conv2 = nn.Conv2d(
            in_channels=cfg.conv1_out_channels,
            out_channels=cfg.conv2_out_channels,
            kernel_size=cfg.conv_kernel_size,
        )
        self.batch2 = nn.BatchNorm2d(num_features=cfg.conv2_out_channels)
        self.max2 = nn.MaxPool2d(
            kernel_size=cfg.pool_kernel_size,
            stride=cfg.pool_stride,
        )
        self.dropout2 = nn.Dropout(p=cfg.dropout2_rate)

        self.fc1 = nn.Linear(in_features=fc1_in, out_features=cfg.fc2_in)
        self.fc2 = nn.Linear(in_features=cfg.fc2_in, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        Args:
            x (torch.Tensor):
                Input batch with shape `(batch_size, C, H, W)`.

        Returns:
            torch.Tensor:
                Class logits with shape `(batch_size, num_classes)`.
        """
        x = self.conv1(x)
        x = F.relu(self.batch1(x))
        x = self.max1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(self.batch2(x))
        x = self.max2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x