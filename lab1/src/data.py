from __future__ import annotations

import logging
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

LOGGER = logging.getLogger(__name__)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


@dataclass(frozen=True)
class DatasetSplits:
    train: Dataset
    validation: Dataset
    test: datasets.CIFAR10
    class_names: list[str]


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    random_crop_padding: int = 4
    rotation_degrees: int = 15
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    use_color_jitter: bool = True


class IndexedCIFAR10Dataset(Dataset):
    """A CIFAR-10 view over selected indices with its own transform."""

    def __init__(
        self,
        base_dataset: datasets.CIFAR10,
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.targets = [base_dataset.targets[index] for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[object, int]:
        dataset_index = self.indices[item]
        image_array = self.base_dataset.data[dataset_index]
        target = self.base_dataset.targets[dataset_index]
        image = Image.fromarray(image_array)
        return self.transform(image), target


def download_and_extract(url: str, save_dir: str, filename: str | None = None) -> str:
    """
    Download a file from ``url`` and extract it when it is an archive.

    The helper is generic so it can be reused for datasets other than CIFAR-10.
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.rstrip("/").split("/")[-1]

    file_path = save_path / filename
    result_path = str(file_path)

    if file_path.exists():
        LOGGER.info(
            "File '%s' already exists in '%s'. Skipping download.",
            filename,
            save_dir,
        )
    else:
        LOGGER.info("Downloading '%s' from '%s'.", filename, url)
        try:
            urllib.request.urlretrieve(url, file_path)
            LOGGER.info("Download successful. Saved to '%s'.", file_path)

            if filename.endswith(".zip"):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(save_path)
                file_path.unlink()
                result_path = str(save_path)
            elif filename.endswith((".tar.gz", ".tgz", ".gz")):
                with tarfile.open(file_path, "r:gz") as tar_ref:
                    tar_ref.extractall(save_path)
                file_path.unlink()
                result_path = str(save_path)
            elif filename.endswith(".tar"):
                with tarfile.open(file_path, "r") as tar_ref:
                    tar_ref.extractall(save_path)
                file_path.unlink()
                result_path = str(save_path)
            else:
                LOGGER.info(
                    "File '%s' is not an archive. No extraction needed.",
                    filename,
                )
        except Exception:
            LOGGER.exception("Failed to download or extract '%s'.", url)
            raise

    return result_path


def _stratified_split_indices(
    targets: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    labels = np.asarray(targets)

    train_indices: list[int] = []
    validation_indices: list[int] = []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        validation_count = max(1, int(len(class_indices) * validation_fraction))
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def download_cifar10(data_dir: str) -> Path:
    """
    Ensure the CIFAR-10 archive is present locally.

    ``torchvision`` can also download the dataset on its own, but this explicit
    stage makes the pipeline's download step visible and reusable.
    """

    extracted_path = download_and_extract(CIFAR10_URL, data_dir)
    return Path(extracted_path)


def build_train_transform(
    augmentation: AugmentationConfig | None = None,
) -> transforms.Compose:
    """Build the training transform pipeline with image augmentation."""

    augmentation = augmentation or AugmentationConfig()

    transform_steps: list[object] = [
        transforms.RandomCrop(32, padding=augmentation.random_crop_padding),
        transforms.RandomHorizontalFlip(p=augmentation.horizontal_flip_probability),
        transforms.RandomRotation(degrees=augmentation.rotation_degrees),
    ]

    if augmentation.use_color_jitter:
        transform_steps.append(
            transforms.ColorJitter(
                brightness=augmentation.color_jitter_brightness,
                contrast=augmentation.color_jitter_contrast,
                saturation=augmentation.color_jitter_saturation,
            )
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    return transforms.Compose(transform_steps)


def build_eval_transform() -> transforms.Compose:
    """Build the deterministic transform pipeline for validation and testing."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )


def build_dataset_splits(
    data_dir: str = "data",
    validation_fraction: float = 0.1,
    seed: int = 42,
    augmentation: AugmentationConfig | None = None,
) -> DatasetSplits:
    """
    Download, ingest, and split CIFAR-10 into train/validation/test datasets.

    The official CIFAR-10 test set is preserved as the test split. The original
    training set is split into train and validation subsets in a class-balanced
    way to avoid noticeably degrading model performance.
    """

    download_cifar10(data_dir)

    train_transform = build_train_transform(augmentation)
    eval_transform = build_eval_transform()

    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=None,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=eval_transform,
    )

    train_indices, validation_indices = _stratified_split_indices(
        targets=full_train_dataset.targets,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    return DatasetSplits(
        train=IndexedCIFAR10Dataset(
            full_train_dataset, train_indices, train_transform
        ),
        validation=IndexedCIFAR10Dataset(
            full_train_dataset, validation_indices, eval_transform
        ),
        test=test_dataset,
        class_names=list(full_train_dataset.classes),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    splits = build_dataset_splits()
    print(f"Train samples: {len(splits.train)}")
    print(f"Validation samples: {len(splits.validation)}")
    print(f"Test samples: {len(splits.test)}")
    print(f"Classes: {splits.class_names}")
