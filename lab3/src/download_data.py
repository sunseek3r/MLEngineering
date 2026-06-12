from __future__ import annotations

import csv
import logging
from pathlib import Path

from config import load_config
from data import download_cifar10

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def write_dataset_registry(
    registry_path: Path,
    source_path: Path,
    sample_count: int,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8", newline="") as registry_file:
        writer = csv.writer(registry_file)
        writer.writerow(
            [
                "dataset_name",
                "version",
                "source_path",
                "tracked_path",
                "description",
                "sample_count",
                "status",
            ]
        )
        writer.writerow(
            [
                "cifar10-batches",
                "v1",
                source_path.as_posix(),
                source_path.as_posix(),
                "Raw CIFAR-10 python batch files for Lab 3 DVC pipeline.",
                sample_count,
                "active",
            ]
        )


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "params.yaml")

    data_root = Path(config.data.data_dir)
    download_cifar10(str(data_root))

    dataset_dir = data_root / "cifar-10-batches-py"
    sample_count = 60000
    registry_path = project_root / "datasets" / "registry" / "dataset_registry.csv"
    write_dataset_registry(registry_path, dataset_dir, sample_count)

    LOGGER.info("Prepared raw dataset at %s", dataset_dir)
    LOGGER.info("Wrote dataset registry to %s", registry_path)


if __name__ == "__main__":
    main()
