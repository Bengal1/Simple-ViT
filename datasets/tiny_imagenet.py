import os
import zipfile
import urllib.request
from PIL import Image
from typing import Callable, Optional

from torch.utils.data import Dataset



class TinyImageNetDataset(Dataset):
    """
    PyTorch-compatible dataset wrapper for Tiny ImageNet.

    Supported splits:
        - "train": training split only
        - "val": validation split only
        - "labeled": all labeled samples from train and validation
        - "test": test split only (labels unavailable)

    Returns:
        tuple[image, label]:
            - For "train", "val", and "labeled": (image, class_index)
            - For "test": (image, -1)
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "labeled",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the Tiny ImageNet dataset.

        Args:
            root (str): Path to the ``tiny-imagenet-200`` dataset directory.
            split (str): Dataset split to load. Must be one of
                ``{"train", "val", "labeled", "test"}``.
            transform (Optional[Callable]): Transform applied to each loaded image.

        Raises:
            ValueError: If ``split`` is not one of the supported options.
        """
        if split not in {"train", "val", "labeled", "test"}:
            raise ValueError("Invalid split!")

        self.split = split
        self.transform = transform

        self.samples: list[tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

        self.root = _download_tiny_imagenet(root)
        self._load_classes()
        self._load_samples()


    def _load_classes(self):
        """
        Load class identifiers from ``wnids.txt`` and build index mappings.

        This ensures a consistent class-to-index mapping across all dataset splits.

        Raises:
            FileNotFoundError: If ``wnids.txt`` does not exist.
        """
        wnids_path = os.path.join(self.root, "wnids.txt")
        if not os.path.isfile(wnids_path):
            raise FileNotFoundError(
                f"Missing Tiny ImageNet class file: {wnids_path}")

        with open(wnids_path, "r") as f:
            wnids = [line.strip() for line in f]

        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        self.idx_to_class = {idx: wnid for wnid, idx in self.class_to_idx.items()}


    def _load_samples(self):
        """
        Load samples according to the selected split.
        """
        if self.split == "labeled":
            self._load_train()
            self._load_val()
        elif self.split == "train":
            self._load_train()
        elif self.split == "val":
            self._load_val()
        elif self.split == "test":
            self._load_test()


    def _load_train(self):
        """
        Load labeled samples from the training split.

        Raises:
            FileNotFoundError: If the training directory does not exist.
        """
        train_dir = os.path.join(self.root, "train")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Missing Tiny ImageNet train directory: {train_dir}")

        for wnid in self.class_to_idx.keys():
            class_dir = os.path.join(train_dir, wnid, "images")

            if not os.path.isdir(class_dir):
                continue

            label = self.class_to_idx[wnid]

            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".JPEG"):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label))


    def _load_val(self):
        """
        Load labeled samples from the validation split using the annotation file.

        Raises:
            FileNotFoundError:
                If the validation images directory or annotation file does not exist.
        """
        val_dir = os.path.join(self.root, "val")
        images_dir = os.path.join(val_dir, "images")
        annotations_file = os.path.join(val_dir, "val_annotations.txt")

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Missing Tiny ImageNet validation images directory: {images_dir}")

        if not os.path.isfile(annotations_file):
            raise FileNotFoundError(
                f"Missing Tiny ImageNet validation annotations file: {annotations_file}")

        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, wnid = parts[0], parts[1]

                if wnid not in self.class_to_idx:
                    continue

                label = self.class_to_idx[wnid]
                path = os.path.join(images_dir, fname)

                self.samples.append((path, label))


    def _load_test(self):
        """
        Load unlabeled samples from the test split.

        Test samples are assigned the placeholder label ``-1``.

        Raises:
            FileNotFoundError: If the test images directory does not exist.
        """
        test_dir = os.path.join(self.root, "test", "images")
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(
                f"Missing Tiny ImageNet test images directory: {test_dir}")

        for fname in sorted(os.listdir(test_dir)):
            if fname.endswith(".JPEG"):
                path = os.path.join(test_dir, fname)
                self.samples.append((path, -1))  # no labels


    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)


    def __getitem__(self, index: int):
        """
        Retrieve a single dataset sample.

        Args:
            index (int): Sample index.

        Returns:
            tuple[image, int]: The transformed image and its label.

        Raises:
            RuntimeError: If the image cannot be loaded.
        """
        path, label = self.samples[index]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path}") from e

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# --- Helper function ---
def _download_tiny_imagenet(
    save_dir: str = "./data",
    url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
) -> str:
    """
    Download and extract the Tiny ImageNet dataset.

    If the dataset already exists in ``save_dir``, the download is skipped.

    Args:
        save_dir (str): Directory where the dataset will be downloaded and extracted.
        url (str): URL to the Tiny ImageNet archive.

    Returns:
        str: Path to the extracted ``tiny-imagenet-200`` directory.

    Raises:
        RuntimeError: If download or extraction fails.
    """

    os.makedirs(save_dir, exist_ok=True)

    zip_path = os.path.join(save_dir, "tiny-imagenet-200.zip")
    dataset_path = os.path.join(save_dir, "tiny-imagenet-200")

    # Already exists → skip
    if os.path.isdir(dataset_path):
        return dataset_path

    try:
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)

        print("Done.")

    except Exception as e:
        raise RuntimeError("Failed to download or extract Tiny ImageNet") from e

    return dataset_path