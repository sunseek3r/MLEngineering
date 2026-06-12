from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn.functional as functional
import yaml
from config import AppConfig, load_config
from data import DatasetSplits, build_dataset_splits
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        conv_channels: tuple[int, int, int],
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c1, c2, c3 = conv_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _cuda_is_usable() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        sample = torch.zeros((1, 3, 32, 32), device="cuda")
        layer = nn.Conv2d(3, 8, kernel_size=3, padding=1).to("cuda")
        _ = layer(sample)
        torch.cuda.synchronize()
        return True
    except RuntimeError as error:
        LOGGER.warning("CUDA was detected but is not usable: %s", error)
        return False


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if _cuda_is_usable():
            return torch.device("cuda")
        LOGGER.info("Falling back to CPU because the current CUDA setup is unusable.")
        return torch.device("cpu")
    if device_name == "cuda" and not _cuda_is_usable():
        raise RuntimeError(
            "CUDA was requested explicitly, but the current PyTorch/CUDA build "
            "cannot run on this GPU. Set `training.device: cpu` or install a "
            "PyTorch build that supports your GPU."
        )
    return torch.device(device_name)


def build_dataloaders(
    config: AppConfig,
    device: torch.device,
) -> tuple[
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
    DatasetSplits,
]:
    splits: DatasetSplits = build_dataset_splits(
        data_dir=config.data.data_dir,
        available_training_batches=list(
            config.data.batch_selection.available_training_batches
        ),
        num_training_batches=config.data.batch_selection.num_training_batches,
        num_validation_batches=config.data.batch_selection.num_validation_batches,
        test_batch_names=list(config.data.batch_selection.test_batch_names),
        validation_fraction=config.data.validation_fraction,
        seed=config.data.seed,
        augmentation=config.data.augmentation,
    )
    LOGGER.info(
        "Using batches | train=%s | validation=%s | test=%s",
        splits.train_batch_names,
        splits.validation_batch_names,
        splits.test_batch_names,
    )

    train_loader = DataLoader(
        splits.train,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        splits.validation,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, validation_loader, test_loader, splits


def build_model(config: AppConfig, device: torch.device) -> nn.Module:
    model = SimpleCNN(
        input_channels=config.model.input_channels,
        num_classes=config.model.num_classes,
        conv_channels=tuple(config.model.conv_channels),
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    optimizer: AdamW,
    device: torch.device,
    epoch_index: int,
    log_every_n_steps: int,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    total_examples = 0
    correct_predictions = 0

    progress = tqdm(data_loader, desc=f"Epoch {epoch_index + 1} [train]", leave=False)
    for step_index, (inputs, targets) in enumerate(progress, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * batch_size
        total_examples += batch_size
        correct_predictions += (predictions == targets).sum().item()

        if step_index % log_every_n_steps == 0:
            LOGGER.info(
                "Epoch %s step %s: training loss %.4f",
                epoch_index + 1,
                step_index,
                running_loss / total_examples,
            )

    return {
        "loss": running_loss / total_examples,
        "accuracy": correct_predictions / total_examples,
    }


def _compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets, predictions, strict=False):
        confusion[target.long(), prediction.long()] += 1

    true_positives = confusion.diag().to(torch.float32)
    false_positives = confusion.sum(dim=0).to(torch.float32) - true_positives
    false_negatives = confusion.sum(dim=1).to(torch.float32) - true_positives

    precision_per_class = true_positives / (
        true_positives + false_positives
    ).clamp_min(1.0)
    recall_per_class = true_positives / (
        true_positives + false_negatives
    ).clamp_min(1.0)
    f1_per_class = 2 * precision_per_class * recall_per_class / (
        precision_per_class + recall_per_class
    ).clamp_min(1e-8)

    accuracy = true_positives.sum().item() / confusion.sum().item()
    return {
        "accuracy": accuracy,
        "precision_macro": precision_per_class.mean().item(),
        "recall_macro": recall_per_class.mean().item(),
        "f1_macro": f1_per_class.mean().item(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    num_classes: int,
    split_name: str,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    progress = tqdm(data_loader, desc=f"[{split_name}]", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = functional.cross_entropy(logits, targets)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    stacked_predictions = torch.cat(all_predictions)
    stacked_targets = torch.cat(all_targets)
    metrics = _compute_classification_metrics(
        predictions=stacked_predictions,
        targets=stacked_targets,
        num_classes=num_classes,
    )
    metrics["loss"] = total_loss / total_examples
    return metrics


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as metrics_file:
        yaml.safe_dump(metrics, metrics_file, sort_keys=False)


def save_best_model(model: nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def _flatten_for_mlflow(
    payload: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        compound_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_for_mlflow(value, compound_key))
        elif isinstance(value, (list, tuple)):
            flattened[compound_key] = ",".join(str(item) for item in value)
        else:
            flattened[compound_key] = value
    return flattened


def _configure_mlflow(config: AppConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)


def _log_config_to_mlflow(config: AppConfig) -> None:
    flattened = _flatten_for_mlflow(asdict(config))
    mlflow.log_params(flattened)


def _log_epoch_metrics_to_mlflow(
    epoch_index: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    step = epoch_index + 1
    for metric_name, metric_value in train_metrics.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value, step=step)
    for metric_name, metric_value in validation_metrics.items():
        mlflow.log_metric(f"validation_{metric_name}", metric_value, step=step)
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value, step=step)


def run_training(config: AppConfig) -> dict[str, Any]:
    if config.mlflow.enabled:
        _configure_mlflow(config)

    active_run = None
    if config.mlflow.enabled and config.mlflow.auto_start_run:
        active_run = mlflow.start_run(run_name=config.mlflow.run_name)
        _log_config_to_mlflow(config)

    device = resolve_device(config.training.device)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Starting data ingestion and split creation.")
    train_loader, validation_loader, test_loader, splits = build_dataloaders(
        config, device
    )

    model = build_model(config, device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    best_validation_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []

    for epoch_index in range(config.training.epochs):
        LOGGER.info("Starting epoch %s/%s", epoch_index + 1, config.training.epochs)
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch_index,
            log_every_n_steps=config.training.log_every_n_steps,
        )
        validation_metrics = evaluate(
            model=model,
            data_loader=validation_loader,
            device=device,
            num_classes=config.model.num_classes,
            split_name="validation",
        )
        test_metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            num_classes=config.model.num_classes,
            split_name="test",
        )

        LOGGER.info(
            (
                "Epoch %s complete | train acc %.4f | "
                "val acc %.4f | val f1 %.4f | test acc %.4f"
            ),
            epoch_index + 1,
            train_metrics["accuracy"],
            validation_metrics["accuracy"],
            validation_metrics["f1_macro"],
            test_metrics["accuracy"],
        )

        epoch_summary = {
            "epoch": epoch_index + 1,
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        history.append(epoch_summary)
        if config.mlflow.enabled and mlflow.active_run() is not None:
            _log_epoch_metrics_to_mlflow(
                epoch_index,
                train_metrics,
                validation_metrics,
                test_metrics,
            )

        if validation_metrics["f1_macro"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1_macro"]
            best_epoch = epoch_index + 1
            best_model_path = (
                Path(config.artifacts.output_dir)
                / config.artifacts.best_model_filename
            )
            save_best_model(model, best_model_path)
            LOGGER.info("Saved new best model to %s", best_model_path)

    metrics_payload = {
        "config": asdict(config),
        "split_summary": {
            "train_batch_names": splits.train_batch_names,
            "validation_batch_names": splits.validation_batch_names,
            "test_batch_names": splits.test_batch_names,
            "train_samples": len(splits.train),
            "validation_samples": len(splits.validation),
            "test_samples": len(splits.test),
        },
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "history": history,
    }
    metrics_path = (
        Path(config.artifacts.output_dir) / config.artifacts.metrics_filename
    )
    save_metrics(metrics_payload, metrics_path)
    LOGGER.info("Saved metrics to %s", metrics_path)

    if config.mlflow.enabled and mlflow.active_run() is not None:
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_validation_f1", best_validation_f1)
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        best_model_path = (
            Path(config.artifacts.output_dir) / config.artifacts.best_model_filename
        )
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path="model")
        params_path = Path(__file__).resolve().parent.parent / "params.yaml"
        if params_path.exists():
            mlflow.log_artifact(str(params_path), artifact_path="config")

    if active_run is not None:
        mlflow.end_run()

    return metrics_payload


def main() -> None:
    configure_logging()
    config = load_config(Path(__file__).resolve().parent.parent / "params.yaml")
    run_training(config)


if __name__ == "__main__":
    main()
