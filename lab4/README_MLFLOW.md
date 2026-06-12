# MLflow Setup For Lab 4

## Tracking Server

- Backend store: `./mlruns`
- Artifact root: `./mlartifacts`
- Default host: `127.0.0.1`
- Default port: `5000`

## Typical Commands

```bash
../lab1/.venv/bin/mlflow server --backend-store-uri mlruns --artifacts-destination mlartifacts --host 127.0.0.1 --port 5000
../lab1/.venv/bin/mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

## UI

When the server is running, open:

`http://127.0.0.1:5000`
