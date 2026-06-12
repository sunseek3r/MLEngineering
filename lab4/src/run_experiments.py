from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlflow
import yaml
from config import AppConfig, BatchSelectionConfig, ExperimentRunConfig, load_config
from train import _log_config_to_mlflow, configure_logging, run_training

LOGGER = logging.getLogger(__name__)


def _build_experiment_config(
    base_config: AppConfig,
    experiment_run: ExperimentRunConfig,
) -> AppConfig:
    batch_selection = BatchSelectionConfig(
        available_training_batches=experiment_run.available_training_batches,
        num_training_batches=experiment_run.num_training_batches,
        num_validation_batches=experiment_run.num_validation_batches,
        test_batch_names=base_config.data.batch_selection.test_batch_names,
        static_test_set=base_config.data.batch_selection.static_test_set,
    )
    data_config = replace(base_config.data, batch_selection=batch_selection)
    mlflow_config = replace(
        base_config.mlflow,
        auto_start_run=False,
        run_name=f"Experiment / Training / {experiment_run.name}",
    )
    artifacts_config = replace(
        base_config.artifacts,
        output_dir=str(Path(base_config.experiments.output_dir) / experiment_run.name),
    )
    return replace(
        base_config,
        data=data_config,
        artifacts=artifacts_config,
        mlflow=mlflow_config,
    )


def _extract_selected_metrics(metrics_payload: dict[str, Any]) -> dict[str, float]:
    best_epoch = metrics_payload["best_epoch"]
    selected_epoch = next(
        epoch_summary
        for epoch_summary in metrics_payload["history"]
        if epoch_summary["epoch"] == best_epoch
    )
    test_metrics = selected_epoch["test"]
    validation_metrics = selected_epoch["validation"]
    return {
        "selected_epoch": best_epoch,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_f1_macro": validation_metrics["f1_macro"],
    }


def _analyze_results(experiment_results: list[dict[str, Any]]) -> list[str]:
    if not experiment_results:
        return ["No experiment results were generated."]

    best_run = max(experiment_results, key=lambda result: result["metrics"]["test_f1_macro"])
    smallest_run = min(
        experiment_results, key=lambda result: result["split_summary"]["train_samples"]
    )
    largest_run = max(
        experiment_results, key=lambda result: result["split_summary"]["train_samples"]
    )

    analysis = [
        (
            f"Best static-test F1 came from '{best_run['name']}' "
            f"with {best_run['metrics']['test_f1_macro']:.4f}."
        ),
        (
            f"The smallest training setup used {smallest_run['split_summary']['train_samples']} "
            f"samples across {smallest_run['split_summary']['train_batch_names']}."
        ),
        (
            f"The largest training setup used {largest_run['split_summary']['train_samples']} "
            f"samples across {largest_run['split_summary']['train_batch_names']}."
        ),
    ]

    if largest_run["name"] != smallest_run["name"]:
        delta_accuracy = (
            largest_run["metrics"]["test_accuracy"]
            - smallest_run["metrics"]["test_accuracy"]
        )
        delta_f1 = (
            largest_run["metrics"]["test_f1_macro"]
            - smallest_run["metrics"]["test_f1_macro"]
        )
        analysis.append(
            "Increasing the training set from "
            f"{smallest_run['split_summary']['train_samples']} to "
            f"{largest_run['split_summary']['train_samples']} samples changed "
            f"test accuracy by {delta_accuracy:+.4f} and test F1 by {delta_f1:+.4f}."
        )

    return analysis


def save_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as summary_file:
        yaml.safe_dump(summary, summary_file, sort_keys=False)


def main() -> None:
    configure_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "params.yaml")

    if not config.experiments.enabled:
        LOGGER.info("Experiments are disabled in params.yaml.")
        return

    if config.mlflow.enabled:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

    experiment_results: list[dict[str, Any]] = []

    for run_index, experiment_run in enumerate(config.experiments.runs, start=1):
        LOGGER.info("Starting experiment '%s'.", experiment_run.name)
        experiment_config = _build_experiment_config(config, experiment_run)
        run_name = (
            f"Experiment {run_index} - Stage 2: Model Training - "
            f"{experiment_run.name}"
        )
        with mlflow.start_run(
            run_name=run_name if config.mlflow.enabled else None
        ):
            if config.mlflow.enabled:
                mlflow.set_tags(
                    {
                        "stage": "model-training",
                        "run_group": "lab4-experiments",
                        "experiment_variant": experiment_run.name,
                    }
                )
                _log_config_to_mlflow(experiment_config)
            metrics_payload = run_training(experiment_config)

        experiment_results.append(
            {
                "name": experiment_run.name,
                "description": experiment_run.description,
                "split_summary": metrics_payload["split_summary"],
                "metrics": _extract_selected_metrics(metrics_payload),
                "best_epoch": metrics_payload["best_epoch"],
                "best_validation_f1": metrics_payload["best_validation_f1"],
                "artifacts_dir": experiment_config.artifacts.output_dir,
            }
        )

    summary = {
        "static_test_batch_names": list(config.data.batch_selection.test_batch_names),
        "experiments": experiment_results,
        "analysis": _analyze_results(experiment_results),
    }
    summary_path = Path(config.experiments.output_dir) / config.experiments.summary_filename
    save_summary(summary, summary_path)
    LOGGER.info("Saved experiment comparison summary to %s", summary_path)
    if config.mlflow.enabled:
        with mlflow.start_run(
            run_name="Experiment Summary - Stage 3: Comparison"
        ):
            mlflow.set_tags(
                {
                    "stage": "experiment-summary",
                    "run_group": "lab4-experiments",
                }
            )
            mlflow.log_artifact(str(summary_path), artifact_path="reports")


if __name__ == "__main__":
    main()
