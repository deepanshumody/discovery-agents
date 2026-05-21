"""Smoke test: the full pipeline runs and writes all expected artifacts."""

from __future__ import annotations

from pathlib import Path

from discovery_agents import ProductDiscoveryPipeline
from discovery_agents.sample_data import SAMPLE_BRIEF, SAMPLE_EVIDENCE


def test_pipeline_runs_and_writes_artifacts(tmp_path: Path) -> None:
    pipeline = ProductDiscoveryPipeline()
    run = pipeline.run(SAMPLE_BRIEF, SAMPLE_EVIDENCE)

    assert run.selected_direction_id is not None
    assert len(run.directions) >= 5
    assert len(run.critiques) == len(run.directions)
    assert run.coding_spec is not None

    pipeline.write_outputs(run, tmp_path)
    for name in [
        "run_summary.md",
        "coding_agent_handoff.md",
        "canvas.html",
        "agent_run.json",
        "agent_trace.md",
    ]:
        assert (tmp_path / name).exists(), f"{name} missing"
