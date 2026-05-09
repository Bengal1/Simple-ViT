# ----------------------------------------------------------------------
# Copyright (c) 2025, Bengal1
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
Dataset transform wrapper.

This module defines `TransformedDataset`, a lightweight wrapper that applies
input and/or target transforms to samples returned by another dataset.

It is useful when the same underlying dataset or subset must be reused with
different transform pipelines, such as separate train and validation transforms.
"""

from typing import Any, Callable

from torch.utils.data import Dataset


__author__ = "Bengal1"
__all__ = ["TransformedDataset"]


# ============================================================
# Transformed Dataset
# ============================================================

class TransformedDataset(Dataset):
    """
    Wrap a dataset and apply optional transforms to its samples.

    Args:
        dataset (Dataset):
            Base dataset to wrap.
        transform (Callable | None, optional):
            Transform applied to input samples.
        target_transform (Callable | None, optional):
            Transform applied to target labels.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Initialize the transformed dataset wrapper.
        """
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        Return the number of samples in the wrapped dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Retrieve a sample and apply optional input/target transforms.

        Args:
            index (int):
                Sample index.

        Returns:
            tuple[Any, Any]:
                Transformed input sample and target label.
        """
        image, label = self.dataset[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label