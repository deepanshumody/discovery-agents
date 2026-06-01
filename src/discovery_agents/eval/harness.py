"""Evaluation harness: run golden cases, score them, and gate regressions."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import RunConfig
from ..pipeline import ProductDiscoveryPipeline
from .cases import GOLDEN_CASES, EvalCase
from .judge import LLMJudge
from .metrics import ops_metrics, quality_metrics

BASELINE_PATH = Path(__file__).with_name("baseline.json")
DEFAULT_TOLERANCE = 0.02


@dataclass
class Scorecard:
    metrics: dict[str, float] = field(default_factory=dict)  # gated quality + judge
    ops: dict[str, float] = field(default_factory=dict)  # reported, not gated

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics, "ops": self.ops}


@dataclass
class RegressionReport:
    passed: bool
    tolerance: float
    regressions: list[dict[str, float]] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


class EvalHarness:
    """Runs the pipeline over golden cases and aggregates metrics."""

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig.from_env()

    def evaluate(self, cases: list[EvalCase] | None = None) -> Scorecard:
        cases = cases or GOLDEN_CASES
        metric_acc: dict[str, list[float]] = defaultdict(list)
        ops_acc: dict[str, list[float]] = defaultdict(list)

        for case in cases:
            pipeline = ProductDiscoveryPipeline(self.config)
            run = pipeline.run(case.brief, case.evidence)
            judged = LLMJudge(pipeline.llm, pipeline.trace).score(run)
            quality = quality_metrics(run, pipeline.trace)
            for key, value in {**quality, **judged}.items():
                metric_acc[key].append(value)
            for key, value in ops_metrics(pipeline.trace).items():
                ops_acc[key].append(value)

        return Scorecard(
            metrics={k: _mean(v) for k, v in metric_acc.items()},
            ops={k: _mean(v) for k, v in ops_acc.items()},
        )

    def check_regression(
        self,
        scorecard: Scorecard,
        baseline: dict[str, Any],
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> RegressionReport:
        regressions: list[dict[str, float]] = []
        for metric, base_value in baseline.get("metrics", {}).items():
            current = scorecard.metrics.get(metric, 0.0)
            if current + tolerance < float(base_value):
                regressions.append(
                    {"metric": metric, "baseline": float(base_value), "current": current}
                )
        return RegressionReport(
            passed=not regressions, tolerance=tolerance, regressions=regressions
        )


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"metrics": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(scorecard: Scorecard, path: Path = BASELINE_PATH) -> None:
    path.write_text(json.dumps(scorecard.to_dict(), indent=2) + "\n", encoding="utf-8")
