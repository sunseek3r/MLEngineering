from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from config import load_config
from train import build_dataloaders, build_model, configure_logging, evaluate, resolve_device

LOGGER = logging.getLogger(__name__)


def save_evaluation_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as report_file:
        yaml.safe_dump(report, report_file, sort_keys=False)


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "params.yaml")

    if config.mlflow.enabled:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

    device = resolve_device(config.training.device)
    _, _, test_loader, splits = build_dataloaders(config, device)
    model = build_model(config, device)

    model_path = Path(config.artifacts.output_dir) / config.artifacts.best_model_filename
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=config.model.num_classes,
        split_name="test",
    )
    report = {
        "model_path": str(model_path),
        "selected_split": config.evaluation.selected_split,
        "minimum_f1_threshold": config.evaluation.minimum_f1_threshold,
        "test_batch_names": splits.test_batch_names,
        "test_samples": len(splits.test),
        "metrics": test_metrics,
        "threshold_passed": (
            test_metrics["f1_macro"] >= config.evaluation.minimum_f1_threshold
        ),
    }

    output_path = Path(config.artifacts.output_dir) / "evaluation.yaml"
    save_evaluation_report(report, output_path)
    LOGGER.info("Saved evaluation report to %s", output_path)

    if config.mlflow.enabled:
        run_name = f"Evaluation - Stage 3: {Path(config.artifacts.output_dir).name}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "stage": "evaluation",
                    "run_group": "lab4-evaluation",
                    "evaluated_model": str(model_path),
                }
            )
            mlflow.log_params(
                {
                    "evaluation.selected_split": config.evaluation.selected_split,
                    "evaluation.minimum_f1_threshold": (
                        config.evaluation.minimum_f1_threshold
                    ),
                    "artifacts.output_dir": config.artifacts.output_dir,
                }
            )
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"evaluation_{metric_name}", metric_value)
            mlflow.log_metric(
                "evaluation_threshold_passed",
                float(report["threshold_passed"]),
            )
            mlflow.log_artifact(str(output_path), artifact_path="reports")
            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="model")


if __name__ == "__main__":
    main()
