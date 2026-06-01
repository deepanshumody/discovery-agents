"""Tests for evaluation metrics, the LLM-as-judge, and the regression gate."""

from __future__ import annotations

from discovery_agents import ProductDiscoveryPipeline
from discovery_agents.eval import (
    EvalHarness,
    LLMJudge,
    Scorecard,
    load_baseline,
    quality_metrics,
)
from discovery_agents.eval.harness import RegressionReport
from discovery_agents.sample_data import SAMPLE_BRIEF, SAMPLE_EVIDENCE


def _fixture_run():
    pipeline = ProductDiscoveryPipeline()
    run = pipeline.run(SAMPLE_BRIEF, SAMPLE_EVIDENCE)
    return run, pipeline.trace, pipeline.llm


def test_quality_metrics_on_a_real_run() -> None:
    run, trace, _ = _fixture_run()
    metrics = quality_metrics(run, trace)
    assert metrics["citation_precision"] == 1.0  # only real evidence is cited
    assert metrics["direction_count"] == 1.0  # 5+ directions
    assert metrics["selection_validity"] == 1.0
    assert metrics["guardrail_pass_rate"] == 1.0  # clean sample passes guards
    assert 0.0 < metrics["evidence_coverage"] <= 1.0
    assert 0.0 < metrics["distinctiveness"] <= 1.0


def test_llm_judge_returns_bounded_deterministic_scores() -> None:
    run, trace, llm = _fixture_run()
    first = LLMJudge(llm, trace).score(run)
    second = LLMJudge(llm, trace).score(run)
    assert set(first) == {"faithfulness", "relevance", "helpfulness"}
    assert first == second  # deterministic under the mock
    assert all(0.0 <= v <= 1.0 for v in first.values())


def test_harness_evaluate_is_deterministic() -> None:
    harness = EvalHarness()
    a = harness.evaluate()
    b = harness.evaluate()
    assert a.metrics == b.metrics  # ops (latency) may differ, metrics must not


def test_regression_gate_passes_against_itself() -> None:
    harness = EvalHarness()
    scorecard = harness.evaluate()
    report = harness.check_regression(scorecard, {"metrics": scorecard.metrics})
    assert isinstance(report, RegressionReport)
    assert report.passed
    assert report.regressions == []


def test_regression_gate_fails_on_drop() -> None:
    harness = EvalHarness()
    scorecard = Scorecard(metrics={"citation_precision": 0.5}, ops={})
    baseline = {"metrics": {"citation_precision": 1.0}}
    report = harness.check_regression(scorecard, baseline, tolerance=0.02)
    assert not report.passed
    assert report.regressions[0]["metric"] == "citation_precision"


def test_load_baseline_returns_metrics_dict() -> None:
    baseline = load_baseline()
    assert "metrics" in baseline
