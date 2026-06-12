from __future__ import annotations

import logging
from pathlib import Path

import yaml
from config import load_config
from data import combine_cifar10_batches

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_split_manifest(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    batch_selection = config.data.batch_selection

    combined_sets = combine_cifar10_batches(
        data_dir=config.data.data_dir,
        available_training_batches=list(batch_selection.available_training_batches),
        num_training_batches=batch_selection.num_training_batches,
        num_validation_batches=batch_selection.num_validation_batches,
        test_batch_names=list(batch_selection.test_batch_names),
    )

    return {
        "data_dir": config.data.data_dir,
        "static_test_set": batch_selection.static_test_set,
        "class_names": combined_sets.class_names,
        "train": {
            "batch_names": combined_sets.train.batch_names,
            "num_samples": combined_sets.train.size,
        },
        "validation": {
            "batch_names": combined_sets.validation.batch_names,
            "num_samples": combined_sets.validation.size,
        },
        "test": {
            "batch_names": combined_sets.test.batch_names,
            "num_samples": combined_sets.test.size,
        },
    }


def save_manifest(manifest: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(manifest, output_file, sort_keys=False)


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"
    manifest = build_split_manifest(config_path)
    output_path = project_root / "artifacts" / "combined_batches.yaml"
    save_manifest(manifest, output_path)
    LOGGER.info("Saved combined batch manifest to %s", output_path)
    LOGGER.info(
        "Combined splits | train=%s | validation=%s | test=%s",
        manifest["train"],
        manifest["validation"],
        manifest["test"],
    )


if __name__ == "__main__":
    main()
