"""MLflow-backed tracker (lazy import; install the ``mlflow`` extra)."""

from __future__ import annotations

from typing import Any


class MLflowTracker:
    """Logs params/metrics/artifacts to an MLflow tracking server or local store."""

    def __init__(
        self,
        experiment: str = "discovery-agents-ml",
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        import mlflow  # lazy import

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._run = mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def close(self) -> None:
        self._mlflow.end_run()
