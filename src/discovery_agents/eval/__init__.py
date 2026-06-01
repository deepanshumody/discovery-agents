"""Evaluation: deterministic metrics, LLM-as-judge, and a regression gate."""

from __future__ import annotations

from .cases import GOLDEN_CASES, EvalCase
from .harness import (
    BASELINE_PATH,
    EvalHarness,
    RegressionReport,
    Scorecard,
    load_baseline,
    save_baseline,
)
from .judge import LLMJudge
from .metrics import ops_metrics, quality_metrics

__all__ = [
    "BASELINE_PATH",
    "EvalCase",
    "EvalHarness",
    "GOLDEN_CASES",
    "LLMJudge",
    "RegressionReport",
    "Scorecard",
    "load_baseline",
    "ops_metrics",
    "quality_metrics",
    "save_baseline",
]
