from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from data import AugmentationConfig


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data"
    validation_fraction: float = 0.1
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 2
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 3
    num_classes: int = 10
    conv_channels: tuple[int, int, int] = (32, 64, 128)
    hidden_dim: int = 256
    dropout: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    log_every_n_steps: int = 100


@dataclass(frozen=True)
class ArtifactConfig:
    output_dir: str = "artifacts"
    best_model_filename: str = "best_model.pt"
    metrics_filename: str = "metrics.yaml"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def _merge_dataclass_defaults(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    data_raw = raw_config.get("data", {})
    model_raw = raw_config.get("model", {})
    training_raw = raw_config.get("training", {})
    artifacts_raw = raw_config.get("artifacts", {})

    augmentation_defaults = AugmentationConfig().__dict__
    augmentation_raw = _merge_dataclass_defaults(
        augmentation_defaults, data_raw.get("augmentation", {})
    )

    data_defaults = DataConfig().__dict__
    data_values = _merge_dataclass_defaults(data_defaults, data_raw)
    data_values["augmentation"] = AugmentationConfig(**augmentation_raw)

    model_values = _merge_dataclass_defaults(ModelConfig().__dict__, model_raw)
    training_values = _merge_dataclass_defaults(
        TrainingConfig().__dict__, training_raw
    )
    artifact_values = _merge_dataclass_defaults(
        ArtifactConfig().__dict__, artifacts_raw
    )

    return AppConfig(
        data=DataConfig(**data_values),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**training_values),
        artifacts=ArtifactConfig(**artifact_values),
    )
