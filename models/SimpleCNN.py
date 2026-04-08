# ----------------------------------------------------------------------
# Copyright (c) 2022, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
===========================
         SimpleCNN
===========================

A compact convolutional neural network for image classification.

Architecture:
    - 2 convolutional blocks:
        Conv2d → BatchNorm → ReLU → MaxPool → Dropout
    - 2 fully connected layers

The model is dataset-agnostic and accepts inputs of shape (C, H, W).
The input dimension of the classifier is computed dynamically at
initialization, enabling use across datasets such as MNIST, CIFAR-10,
and Tiny ImageNet.

Outputs raw logits and is intended for use with
`torch.nn.CrossEntropyLoss`.
"""
__author__="Bengal1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import CNNConfig


#--------------- Model Definition ---------------#
class SimpleCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network for handwritten digit classification on MNIST.

    This model consists of two convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool → Dropout),
    followed by two fully connected layers. It outputs raw logits and is intended
    to be used with `nn.CrossEntropyLoss`, which applies softmax internally.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        batch1 (nn.BatchNorm2d): Batch normalization after the first conv layer.
        max1 (nn.MaxPool2d): Max pooling after the first conv block.
        dropout1 (nn.Dropout): Dropout after the first pooling layer.

        conv2 (nn.Conv2d): Second convolutional layer.
        batch2 (nn.BatchNorm2d): Batch normalization after the second conv layer.
        max2 (nn.MaxPool2d): Max pooling after the second conv block.
        dropout2 (nn.Dropout): Dropout after the second pooling layer.

        fc1 (nn.Linear): First fully connected layer (dense).
        fc2 (nn.Linear): Output layer mapping to class logits.
    """

    def __init__(
            self,
            input_shape: tuple[int, int, int],
            num_classes: int = 10,
            cfg: CNNConfig = CNNConfig()
    ):
        """
        Initialize the SimpleCNN model.

        Args:
            input_shape (tuple[int, int, int]): Input shape (C, H, W).
            num_classes (int): Number of output classes.
            cfg (CNNConfig): Model configuration.

        Raises:
            ValueError: If input dimensions are too small.
        """
        super().__init__()

        input_channels, h, w = input_shape

        # ---- Compute fc1_in dynamically ---- #
        for _ in range(2):
            # Conv (no padding, stride=1)
            h = h - cfg.conv_kernel_size + 1
            w = w - cfg.conv_kernel_size + 1

            # Pool
            h = (h - cfg.pool_kernel_size) // cfg.pool_stride + 1
            w = (w - cfg.pool_kernel_size) // cfg.pool_stride + 1

        if h <= 0 or w <= 0:
            raise ValueError("Input too small for given architecture")

        fc1_in = cfg.conv2_out_channels * h * w

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=cfg.conv1_out_channels,
                               kernel_size=cfg.conv_kernel_size)
        self.conv2 = nn.Conv2d(in_channels=cfg.conv1_out_channels,
                               out_channels=cfg.conv2_out_channels,
                               kernel_size=cfg.conv_kernel_size)

        # Max-Pooling layers
        self.max1 = nn.MaxPool2d(kernel_size=cfg.pool_kernel_size, stride=cfg.pool_stride)
        self.max2 = nn.MaxPool2d(kernel_size=cfg.pool_kernel_size, stride=cfg.pool_stride)

        # Fully-Connected layers
        self.fc1 = nn.Linear(in_features=fc1_in, out_features=cfg.fc2_in)
        self.fc2 = nn.Linear(in_features=cfg.fc2_in, out_features=num_classes)

        # Dropout
        self.dropout1 = nn.Dropout(p=cfg.dropout1_rate)
        self.dropout2 = nn.Dropout(p=cfg.dropout2_rate)

        # Batch Normalization
        self.batch1 = nn.BatchNorm2d(num_features=cfg.conv1_out_channels)
        self.batch2 = nn.BatchNorm2d(num_features=cfg.conv2_out_channels)

    def forward(self, x):
        """
        Forward pass of the network.
        Note: CrossEntropyLoss handles softmax

        Args:
            x (torch.Tensor): Input batch of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.conv1(x)           # Convolution Layer 1
        x = F.relu(self.batch1(x))  # Batch Normalization + ReLU
        x = self.max1(x)            # Max Pooling
        x = self.dropout1(x)        # Dropout

        x = self.conv2(x)           # Convolution Layer 2
        x = F.relu(self.batch2(x))  # Batch Normalization + ReLU
        x = self.max2(x)            # Max Pooling
        x = self.dropout2(x)        # Dropout

        x = torch.flatten(x, start_dim=1)  # Flatten for FC layer
        x = F.relu(self.fc1(x))     # Fully Connected Layer 1 + ReLU
        x = self.fc2(x)             # Fully Connected Layer 2 (logits)
        return x
