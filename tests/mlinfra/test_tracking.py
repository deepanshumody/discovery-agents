"""Tests for the JSON experiment tracker (dependency-free; runs everywhere)."""

from __future__ import annotations

import json

from discovery_agents.mlinfra.tracking import JSONTracker, Tracker


def test_jsontracker_persists_params_metrics_artifacts(tmp_path) -> None:
    tracker = JSONTracker(run_dir=str(tmp_path / "runs"), run_name="exp1")
    assert isinstance(tracker, Tracker)

    tracker.log_params({"lr": 0.001, "steps": 100})
    tracker.log_metrics({"loss": 1.2}, step=1)
    tracker.log_metrics({"loss": 0.9}, step=2)

    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"weights")
    tracker.log_artifact(str(artifact))

    run_dir = tmp_path / "runs" / "exp1"
    params = json.loads((run_dir / "params.json").read_text())
    assert params == {"lr": 0.001, "steps": 100}

    metric_lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metric_lines) == 2
    assert json.loads(metric_lines[1]) == {"step": 2, "loss": 0.9}

    assert (run_dir / "artifacts" / "weights.bin").exists()


def test_log_params_merges(tmp_path) -> None:
    tracker = JSONTracker(run_dir=str(tmp_path / "runs"), run_name="exp2")
    tracker.log_params({"a": 1})
    tracker.log_params({"b": 2})
    params = json.loads((tmp_path / "runs" / "exp2" / "params.json").read_text())
    assert params == {"a": 1, "b": 2}
