from __future__ import annotations

import logging
import pickle
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


@dataclass(frozen=True)
class DatasetSplits:
    train: Dataset
    validation: Dataset
    test: Dataset
    class_names: list[str]
    train_batch_names: list[str]
    validation_batch_names: list[str]
    test_batch_names: list[str]


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    random_crop_padding: int = 4
    rotation_degrees: int = 15
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    use_color_jitter: bool = True


class ArrayBackedCIFAR10Dataset(Dataset):
    """Dataset backed by already combined CIFAR-10 arrays."""

    def __init__(
        self,
        data: np.ndarray,
        targets: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, item: int) -> tuple[object, int]:
        image = Image.fromarray(self.data[item])
        return self.transform(image), self.targets[item]


@dataclass(frozen=True)
class CombinedBatchSplit:
    data: np.ndarray
    targets: list[int]
    batch_names: list[str]

    @property
    def size(self) -> int:
        return len(self.targets)


@dataclass(frozen=True)
class CombinedBatchSets:
    train: CombinedBatchSplit
    validation: CombinedBatchSplit
    test: CombinedBatchSplit
    class_names: list[str]


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


def download_cifar10(data_dir: str) -> Path:
    """
    Ensure the CIFAR-10 archive is present locally.

    ``torchvision`` can also download the dataset on its own, but this explicit
    stage makes the pipeline's download step visible and reusable.
    """

    extracted_path = download_and_extract(CIFAR10_URL, data_dir)
    return Path(extracted_path)


def _resolve_cifar_batch_dir(data_dir: str) -> Path:
    download_cifar10(data_dir)
    batch_dir = Path(data_dir) / "cifar-10-batches-py"
    if not batch_dir.exists():
        raise FileNotFoundError(f"CIFAR-10 batch directory not found: {batch_dir}")
    return batch_dir


def _load_batch_file(batch_path: Path) -> tuple[np.ndarray, list[int]]:
    with batch_path.open("rb") as batch_file:
        payload = pickle.load(batch_file, encoding="bytes")

    data = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    targets = list(payload[b"labels"])
    return data, targets


def _load_class_names(batch_dir: Path) -> list[str]:
    meta_path = batch_dir / "batches.meta"
    with meta_path.open("rb") as meta_file:
        payload = pickle.load(meta_file, encoding="bytes")
    return [label.decode("utf-8") for label in payload[b"label_names"]]


def _batch_names_from_ids(batch_ids: list[int]) -> list[str]:
    return [f"data_batch_{batch_id}" for batch_id in batch_ids]


def _combine_named_batches(batch_dir: Path, batch_names: list[str]) -> CombinedBatchSplit:
    batch_arrays: list[np.ndarray] = []
    combined_targets: list[int] = []

    for batch_name in batch_names:
        batch_path = batch_dir / batch_name
        if not batch_path.exists():
            raise FileNotFoundError(f"Batch file not found: {batch_path}")
        batch_data, batch_targets = _load_batch_file(batch_path)
        batch_arrays.append(batch_data)
        combined_targets.extend(batch_targets)

    if not batch_arrays:
        raise ValueError("At least one batch must be selected for each split.")

    return CombinedBatchSplit(
        data=np.concatenate(batch_arrays, axis=0),
        targets=combined_targets,
        batch_names=batch_names,
    )


def combine_cifar10_batches(
    data_dir: str,
    available_training_batches: list[int],
    num_training_batches: int,
    num_validation_batches: int,
    test_batch_names: list[str] | None = None,
) -> CombinedBatchSets:
    batch_dir = _resolve_cifar_batch_dir(data_dir)
    class_names = _load_class_names(batch_dir)

    if len(set(available_training_batches)) != len(available_training_batches):
        raise ValueError("available_training_batches must not contain duplicates.")
    if num_training_batches <= 0:
        raise ValueError("num_training_batches must be greater than zero.")
    if num_validation_batches <= 0:
        raise ValueError("num_validation_batches must be greater than zero.")
    if num_training_batches + num_validation_batches > len(available_training_batches):
        raise ValueError(
            "The requested training and validation batch counts exceed the number "
            "of available training batches."
        )

    training_batch_ids = available_training_batches[:num_training_batches]
    validation_start = num_training_batches
    validation_end = validation_start + num_validation_batches
    validation_batch_ids = available_training_batches[validation_start:validation_end]

    selected_test_batch_names = test_batch_names or ["test_batch"]

    return CombinedBatchSets(
        train=_combine_named_batches(batch_dir, _batch_names_from_ids(training_batch_ids)),
        validation=_combine_named_batches(
            batch_dir,
            _batch_names_from_ids(validation_batch_ids),
        ),
        test=_combine_named_batches(batch_dir, selected_test_batch_names),
        class_names=class_names,
    )


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
    available_training_batches: list[int] | None = None,
    num_training_batches: int = 3,
    num_validation_batches: int = 1,
    test_batch_names: list[str] | None = None,
    validation_fraction: float = 0.1,
    seed: int = 42,
    augmentation: AugmentationConfig | None = None,
) -> DatasetSplits:
    """
    Download, ingest, and split CIFAR-10 into train/validation/test datasets.

    The test split remains static via the configured CIFAR-10 test batch. Training
    and validation are built dynamically by combining the configured training
    batch files from the larger CIFAR-10 training set.
    """

    train_transform = build_train_transform(augmentation)
    eval_transform = build_eval_transform()
    _ = validation_fraction
    _ = seed

    combined_sets = combine_cifar10_batches(
        data_dir=data_dir,
        available_training_batches=available_training_batches or [1, 2, 3, 4, 5],
        num_training_batches=num_training_batches,
        num_validation_batches=num_validation_batches,
        test_batch_names=test_batch_names,
    )

    return DatasetSplits(
        train=ArrayBackedCIFAR10Dataset(
            combined_sets.train.data,
            combined_sets.train.targets,
            train_transform,
        ),
        validation=ArrayBackedCIFAR10Dataset(
            combined_sets.validation.data,
            combined_sets.validation.targets,
            eval_transform,
        ),
        test=ArrayBackedCIFAR10Dataset(
            combined_sets.test.data,
            combined_sets.test.targets,
            eval_transform,
        ),
        class_names=combined_sets.class_names,
        train_batch_names=combined_sets.train.batch_names,
        validation_batch_names=combined_sets.validation.batch_names,
        test_batch_names=combined_sets.test.batch_names,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    splits = build_dataset_splits()
    print(f"Train samples: {len(splits.train)}")
    print(f"Validation samples: {len(splits.validation)}")
    print(f"Test samples: {len(splits.test)}")
    print(f"Classes: {splits.class_names}")
