"""Experiment tracking: a Tracker protocol + a dependency-free JSONTracker default."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    def log_artifact(self, path: str) -> None: ...


class JSONTracker:
    """Writes params/metrics/artifacts to ``<run_dir>/<run_name>/`` — no dependencies."""

    def __init__(self, run_dir: str = "outputs/ml/runs", run_name: str = "run") -> None:
        self.dir = Path(run_dir) / run_name
        self.artifacts = self.dir / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self._params_path = self.dir / "params.json"
        self._metrics_path = self.dir / "metrics.jsonl"

    def log_params(self, params: dict[str, Any]) -> None:
        existing: dict[str, Any] = {}
        if self._params_path.exists():
            existing = json.loads(self._params_path.read_text(encoding="utf-8"))
        existing.update(params)
        self._params_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        line = json.dumps({"step": step, **metrics})
        with self._metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def log_artifact(self, path: str) -> None:
        shutil.copy(path, self.artifacts / Path(path).name)
