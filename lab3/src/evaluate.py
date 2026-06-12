from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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


if __name__ == "__main__":
    main()
