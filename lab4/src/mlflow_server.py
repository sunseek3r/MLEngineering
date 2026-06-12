from __future__ import annotations

from pathlib import Path


def build_mlflow_server_command(project_root: Path) -> str:
    backend_store = project_root / "mlruns"
    artifact_root = project_root / "mlartifacts"
    return (
        "../lab1/.venv/bin/mlflow server "
        f"--backend-store-uri {backend_store} "
        f"--artifacts-destination {artifact_root} "
        "--host 127.0.0.1 "
        "--port 5000"
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    print(build_mlflow_server_command(project_root))


if __name__ == "__main__":
    main()
