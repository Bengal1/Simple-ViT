from typing import Callable

from torch.utils.data import Dataset


# ============================================================
# Transformed Dataset
# ============================================================

class TransformedDataset(Dataset):
    """
    Dataset wrapper that applies transforms to samples retrieved
    from another dataset.

    This wrapper enables applying different transforms to the same
    underlying dataset instance without duplicating dataset loading
    or metadata.

    Common use cases include:
        - Different train/validation transforms
        - Additional augmentation pipelines
        - Label transformations

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

    def __getitem__(self, index: int):
        """
        Retrieve and transform a dataset sample.

        The sample is first retrieved from the wrapped dataset, then
        optional input and target transforms are applied independently.

        Args:
            index (int):
                Sample index.

        Returns:
            tuple:
                Transformed input sample and target label.
        """
        # Retrieve raw sample from the wrapped dataset.
        image, label = self.dataset[index]

        # Apply input transformation if provided.
        if self.transform is not None:
            image = self.transform(image)

        # Apply target transformation if provided.
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label