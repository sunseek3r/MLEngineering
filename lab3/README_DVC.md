# DVC Setup For Lab 3

## What Is Tracked

- Raw CIFAR-10 files in `datasets/raw/cifar-10-batches-py`
- Dataset registry table in `datasets/registry/dataset_registry.csv`

## Typical Commands

```bash
dvc repro
dvc status
dvc push
dvc pull
```

## Local Remote

The local remote is configured as `localstorage` and points to `./dvcstore`.

## Pipeline Stages

- `download_data`: downloads the raw CIFAR-10 files and writes the dataset registry CSV
- `train_model`: trains the CNN and saves the best model plus training metrics
- `evaluate_model`: loads the saved model and writes final test metrics

## Parameter Management

The pipeline parameters are managed in `params.yaml`.
Changing values such as dataset paths, batch selection, model hyperparameters,
training settings, or evaluation thresholds will cause `dvc repro` to rerun the
affected stages.
