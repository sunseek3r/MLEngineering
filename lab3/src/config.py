from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from data import AugmentationConfig


@dataclass(frozen=True)
class BatchSelectionConfig:
    available_training_batches: tuple[int, ...] = (1, 2, 3, 4, 5)
    num_training_batches: int = 3
    num_validation_batches: int = 1
    test_batch_names: tuple[str, ...] = ("test_batch",)
    static_test_set: bool = True


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data"
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 2
    validation_fraction: float = 0.1
    batch_selection: BatchSelectionConfig = field(default_factory=BatchSelectionConfig)
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
class EvaluationConfig:
    selected_split: str = "test"
    minimum_f1_threshold: float = 0.7


@dataclass(frozen=True)
class ExperimentRunConfig:
    name: str
    description: str = ""
    available_training_batches: tuple[int, ...] = (1, 2, 3, 4, 5)
    num_training_batches: int = 3
    num_validation_batches: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    enabled: bool = True
    output_dir: str = "artifacts/experiments"
    summary_filename: str = "comparison_summary.yaml"
    runs: tuple[ExperimentRunConfig, ...] = (
        ExperimentRunConfig(
            name="train_1_batch",
            description="Single training batch with a larger validation allocation.",
            available_training_batches=(1, 2, 3, 4, 5),
            num_training_batches=1,
            num_validation_batches=2,
        ),
        ExperimentRunConfig(
            name="train_2_batches",
            description="Two training batches to compare against the smaller setup.",
            available_training_batches=(1, 2, 3, 4, 5),
            num_training_batches=2,
            num_validation_batches=2,
        ),
        ExperimentRunConfig(
            name="train_3_batches",
            description="Three training batches for a larger dynamic training set.",
            available_training_batches=(1, 2, 3, 4, 5),
            num_training_batches=3,
            num_validation_batches=2,
        ),
    )


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)


def _merge_dataclass_defaults(
    defaults: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def _build_experiment_run_config(
    run_raw: ExperimentRunConfig | dict[str, Any],
) -> ExperimentRunConfig:
    if isinstance(run_raw, ExperimentRunConfig):
        return run_raw

    if "name" not in run_raw:
        raise ValueError("Each experiment run must define a name.")

    defaults = ExperimentRunConfig(name=run_raw["name"]).__dict__
    return ExperimentRunConfig(**_merge_dataclass_defaults(defaults, run_raw))


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    data_raw = raw_config.get("data", {})
    model_raw = raw_config.get("model", {})
    training_raw = raw_config.get("training", {})
    artifacts_raw = raw_config.get("artifacts", {})
    evaluation_raw = raw_config.get("evaluation", {})
    experiments_raw = raw_config.get("experiments", {})

    augmentation_defaults = AugmentationConfig().__dict__
    augmentation_raw = _merge_dataclass_defaults(
        augmentation_defaults, data_raw.get("augmentation", {})
    )

    batch_selection_defaults = BatchSelectionConfig().__dict__
    batch_selection_raw = _merge_dataclass_defaults(
        batch_selection_defaults, data_raw.get("batch_selection", {})
    )

    data_defaults = DataConfig().__dict__
    data_values = _merge_dataclass_defaults(data_defaults, data_raw)
    data_values["augmentation"] = AugmentationConfig(**augmentation_raw)
    data_values["batch_selection"] = BatchSelectionConfig(**batch_selection_raw)

    model_values = _merge_dataclass_defaults(ModelConfig().__dict__, model_raw)
    training_values = _merge_dataclass_defaults(
        TrainingConfig().__dict__, training_raw
    )
    artifact_values = _merge_dataclass_defaults(
        ArtifactConfig().__dict__, artifacts_raw
    )
    evaluation_values = _merge_dataclass_defaults(
        EvaluationConfig().__dict__, evaluation_raw
    )
    experiment_defaults = ExperimentConfig()
    experiment_runs_raw = experiments_raw.get("runs", experiment_defaults.runs)
    experiment_values = _merge_dataclass_defaults(
        {
            "enabled": experiment_defaults.enabled,
            "output_dir": experiment_defaults.output_dir,
            "summary_filename": experiment_defaults.summary_filename,
        },
        {key: value for key, value in experiments_raw.items() if key != "runs"},
    )
    experiment_values["runs"] = tuple(
        _build_experiment_run_config(run_raw) for run_raw in experiment_runs_raw
    )

    return AppConfig(
        data=DataConfig(**data_values),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**training_values),
        artifacts=ArtifactConfig(**artifact_values),
        evaluation=EvaluationConfig(**evaluation_values),
        experiments=ExperimentConfig(**experiment_values),
    )
