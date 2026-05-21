"""Command line entry point.

Usage:
    python -m discovery_agents.cli --output outputs/demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import ProductDiscoveryPipeline
from .sample_data import SAMPLE_BRIEF, SAMPLE_EVIDENCE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the product discovery agent demo.")
    parser.add_argument("--output", default="outputs/demo", help="Directory for generated artifacts.")
    args = parser.parse_args()

    pipeline = ProductDiscoveryPipeline()
    run = pipeline.run(SAMPLE_BRIEF, SAMPLE_EVIDENCE)
    output_dir = Path(args.output)
    pipeline.write_outputs(run, output_dir)

    print(f"Generated agent run in: {output_dir.resolve()}")
    print(f"Selected direction: {run.selected_direction_id}")
    print("Artifacts:")
    for file_name in ["run_summary.md", "coding_agent_handoff.md", "canvas.html", "agent_run.json", "agent_trace.md"]:
        print(f"- {output_dir / file_name}")


if __name__ == "__main__":
    main()
