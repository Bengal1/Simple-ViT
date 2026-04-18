from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from datasets import TinyImageNetDataset


# ============================================================
# MNIST
# ============================================================

def _get_mnist_dataloaders(
    batch_size: int,
    train_validation_split: float,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[int, int, int], int]:
    """
    Create DataLoaders for MNIST.

    The training split is divided into training and validation subsets,
    while the official test split is used unchanged.

    Args:
        batch_size (int): Number of samples per batch.
        train_validation_split (float): Fraction of the training set used
            for validation.

    Returns:
        tuple:
            (train_loader, val_loader, test_loader, image_size, num_classes)

    Raises:
        ValueError: If ``train_validation_split`` is not in (0, 1).
    """
    if not 0 < train_validation_split < 1:
        raise ValueError("train_validation_split must be between 0 and 1.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [1 - train_validation_split, train_validation_split],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    img_size, num_classes = get_dataset_info(train_loader)

    return train_loader, val_loader, test_loader, img_size, num_classes


# ============================================================
# CIFAR-10
# ============================================================

def _get_cifar10_dataloaders(
    batch_size: int,
    train_validation_split: float,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[int, int, int], int]:
    """
    Create DataLoaders for CIFAR-10.

    The training split is divided into training and validation subsets,
    while the official test split is used unchanged.

    Args:
        batch_size (int): Number of samples per batch.
        train_validation_split (float): Fraction of the training set used
            for validation.

    Returns:
        tuple:
            (train_loader, val_loader, test_loader, image_size, num_classes)

    Raises:
        ValueError: If ``train_validation_split`` is not in (0, 1).
    """
    if not 0 < train_validation_split < 1:
        raise ValueError("train_validation_split must be between 0 and 1.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    full_train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [1 - train_validation_split, train_validation_split],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    img_size, num_classes = get_dataset_info(train_loader)

    return train_loader, val_loader, test_loader, img_size, num_classes


# ============================================================
# Tiny ImageNet
# ============================================================

def _get_tiny_imagenet_dataloaders(
    batch_size: int,
    train_validation_split: float,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[int, int, int], int]:
    """
    Create DataLoaders for Tiny ImageNet.

    The labeled dataset (train + val) is split into training and validation
    subsets, while the test split is used unchanged.

    Args:
        batch_size (int): Number of samples per batch.
        train_validation_split (float): Fraction of the labeled dataset used
            for validation.

    Returns:
        tuple:
            (train_loader, val_loader, test_loader, image_size, num_classes)

    Raises:
        ValueError: If ``train_validation_split`` is not in (0, 1).
    """
    if not 0 < train_validation_split < 1:
        raise ValueError("train_validation_split must be between 0 and 1.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4802, 0.4481, 0.3975),
            (0.2302, 0.2265, 0.2262),
        ),
    ])

    full_train_dataset = TinyImageNetDataset(
        root="./data",
        split="train",
        transform=transform,
    )

    test_dataset = TinyImageNetDataset(
        root="./data",
        split="val",
        transform=transform,
    )

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [1 - train_validation_split, train_validation_split],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    img_size, num_classes = get_dataset_info(train_loader)

    return train_loader, val_loader, test_loader, img_size, num_classes


# ============================================================
# Public API
# ============================================================

def get_dataloaders(
    dataset: str = "mnist",
    batch_size: int = 128,
    train_validation_split: float = 0.2,
) -> tuple[DataLoader, DataLoader, DataLoader, tuple[int, int, int], int]:
    """
    Return DataLoaders for the selected dataset.

    Supported datasets:
        - "mnist"
        - "cifar10"
        - "tiny_imagenet"

    Args:
        dataset (str, optional): Dataset name. Default is "mnist".
        batch_size (int, optional): Batch size. Default is 128.
        train_validation_split (float, optional): Fraction used for validation.
            Default is 0.2.

    Returns:
        tuple:
            (train_loader, val_loader, test_loader, image_size, num_classes)

    Raises:
        ValueError: If the dataset is not supported.
    """
    dataset = dataset.lower()

    if dataset == "mnist":
        return _get_mnist_dataloaders(batch_size, train_validation_split)

    if dataset == "cifar10":
        return _get_cifar10_dataloaders(batch_size, train_validation_split)

    if dataset == "tiny_imagenet":
        return _get_tiny_imagenet_dataloaders(batch_size, train_validation_split)

    raise ValueError(
        f"Unsupported dataset '{dataset}'. "
        "Choose from: 'mnist', 'cifar10', 'tiny_imagenet'."
    )


# ============================================================
# DataLoader Helper Utilities
# ============================================================

def get_dataset_info(
    data_loader: DataLoader,
) -> tuple[tuple[int, int, int], int]:
    """
    Extract the input image shape and number of classes from a DataLoader.

    This function inspects a single batch to determine the image shape
    `(C, H, W)` and resolves the underlying dataset to infer the total
    number of classes. It supports standard torchvision datasets and
    wrapped datasets such as `torch.utils.data.Subset`.

    Args:
        data_loader (DataLoader):
            DataLoader associated with an image classification dataset.

    Returns:
        tuple[tuple[int, int, int], int]:
            A tuple containing:
                - image_size: Input image shape as `(C, H, W)`
                - num_classes: Total number of classes in the dataset

    Raises:
        ValueError:
            If the DataLoader is empty or the number of classes cannot
            be determined from the dataset.
    """
    try:
        images, _ = next(iter(data_loader))
    except StopIteration as exc:
        raise ValueError("Cannot extract dataset info from an empty DataLoader.") from exc

    if images.ndim != 4:
        raise ValueError(
            f"Expected image batch with shape [B, C, H, W], got {tuple(images.shape)}."
        )

    _, channels, height, width = images.shape
    image_size = (channels, height, width)

    dataset = data_loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    if hasattr(dataset, "classes"):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, "targets"):
        num_classes = len(set(dataset.targets))
    else:
        raise ValueError(
            "Unable to determine the number of classes from the dataset. "
            "Expected a dataset with either `classes` or `targets`."
        )

    return image_size, num_classes