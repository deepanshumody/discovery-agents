"""Command line entry point.

Examples:
    python -m discovery_agents.cli --output outputs/demo
    python -m discovery_agents.cli --provider anthropic --model claude-sonnet-4-6
    python -m discovery_agents.cli --eval                 # run the eval regression gate
    python -m discovery_agents.cli --eval --update-baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import RunConfig
from .pipeline import ProductDiscoveryPipeline
from .sample_data import SAMPLE_BRIEF, SAMPLE_EVIDENCE


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the product discovery agent workflow.")
    parser.add_argument("--output", default="outputs/demo", help="Directory for artifacts.")
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider: mock (default) | anthropic | cohere | openai.",
    )
    parser.add_argument("--model", default=None, help="Override the provider's default model.")
    parser.add_argument(
        "--no-rag", action="store_true", help="Disable retrieval grounding for this run."
    )
    parser.add_argument(
        "--use-langgraph", action="store_true", help="Execute the graph via LangGraph if installed."
    )
    parser.add_argument(
        "--eval", action="store_true", help="Run the evaluation harness + regression gate."
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="With --eval, overwrite the committed baseline with the current scores.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig.from_env()
    if args.provider:
        config.provider = args.provider.lower()
    if args.model:
        config.model = args.model
    if args.no_rag:
        config.use_rag = False
    if args.use_langgraph:
        config.use_langgraph = True
    return config


def _run_eval(config: RunConfig, *, update_baseline: bool) -> int:
    from .eval import EvalHarness, load_baseline, save_baseline

    harness = EvalHarness(config)
    scorecard = harness.evaluate()

    print("Eval scorecard (quality metrics):")
    for metric, value in sorted(scorecard.metrics.items()):
        print(f"  {metric:24s} {value:.4f}")
    print("Ops (reported, not gated):")
    for metric, value in sorted(scorecard.ops.items()):
        print(f"  {metric:24s} {value:.4f}")

    if update_baseline:
        save_baseline(scorecard)
        print("Baseline updated.")
        return 0

    report = harness.check_regression(scorecard, load_baseline())
    if report.passed:
        print(f"Regression gate PASSED (tolerance {report.tolerance}).")
        return 0
    print("Regression gate FAILED:")
    for reg in report.regressions:
        print(f"  {reg['metric']}: baseline {reg['baseline']:.4f} -> current {reg['current']:.4f}")
    return 1


def _run_demo(config: RunConfig, output: str) -> int:
    pipeline = ProductDiscoveryPipeline(config)
    run = pipeline.run(SAMPLE_BRIEF, SAMPLE_EVIDENCE)
    output_dir = Path(output)
    pipeline.write_outputs(run, output_dir)

    print(f"Generated agent run in: {output_dir.resolve()}")
    print(f"Provider: {config.provider} ({config.resolved_model()})")
    print(f"Selected direction: {run.selected_direction_id}")
    print(
        f"Trace: {len(pipeline.trace.spans)} spans, "
        f"{pipeline.trace.total_usage.total_tokens} tokens, "
        f"${pipeline.trace.total_cost_usd:.6f}"
    )
    print("Artifacts:")
    for file_name in [
        "run_summary.md",
        "coding_agent_handoff.md",
        "canvas.html",
        "agent_run.json",
        "agent_trace.md",
    ]:
        print(f"- {output_dir / file_name}")
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    config = _config_from_args(args)
    if args.eval:
        sys.exit(_run_eval(config, update_baseline=args.update_baseline))
    sys.exit(_run_demo(config, args.output))


if __name__ == "__main__":
    main()
