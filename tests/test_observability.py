"""Tests for trace spans, cost accounting, and roll-ups."""

from __future__ import annotations

from discovery_agents.llm.types import LLMResponse, TokenUsage
from discovery_agents.observability import cost_for, render_trace_html
from discovery_agents.observability.trace import Trace, TraceSpan


def test_cost_for_known_model() -> None:
    # claude-sonnet-4-6: (0.003, 0.015) per 1k tokens.
    usage = TokenUsage(input_tokens=1000, output_tokens=1000)
    assert cost_for("claude-sonnet-4-6", usage) == round(0.003 + 0.015, 6)


def test_cost_for_mock_is_free() -> None:
    usage = TokenUsage(input_tokens=5000, output_tokens=5000)
    assert cost_for("mock-1", usage) == 0.0


def test_record_llm_computes_cost_and_rolls_up() -> None:
    trace = Trace()
    resp = LLMResponse(
        text="ok",
        usage=TokenUsage(input_tokens=1000, output_tokens=1000),
        latency_ms=12.5,
        model="claude-sonnet-4-6",
    )
    span = trace.record_llm("IdeationAgent", "ideation.generate", resp)
    assert span.cost_usd == round(0.018, 6)
    assert trace.total_cost_usd == round(0.018, 6)
    assert trace.total_usage.total_tokens == 2000
    assert trace.total_latency_ms == 12.5


def test_log_back_compat_and_markdown() -> None:
    trace = Trace()
    trace.log("StrategyAgent", "did a thing", count=3)
    md = trace.as_markdown()
    assert "StrategyAgent" in md
    assert "Total spans: 1" in md


def test_render_trace_html_contains_totals() -> None:
    trace = Trace()
    trace.record(TraceSpan(agent="A", op="op", model="mock-1"))
    html = render_trace_html(trace)
    assert "Run trace" in html
    assert "<table>" in html
