"""Structured, auditable tracing for multi-agent runs.

Every meaningful step (an LLM call, a tool call, a guardrail decision, or a plain
log event) is recorded as a `TraceSpan` carrying latency, token usage, and cost.
The trace is the audit log the JD calls for ("observable, safe, auditable").

`Trace` keeps the old `AgentTrace.log(...)`/`as_markdown()` surface so existing
agents need no change, while adding `record(...)` and roll-up totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..llm.types import LLMResponse, TokenUsage
from .cost import cost_for


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceSpan:
    """One recorded step in a run."""

    agent: str
    op: str = "log"
    message: str = ""
    latency_ms: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    model: str = ""
    tool_calls: list[str] = field(default_factory=list)
    guardrail_events: list[dict[str, Any]] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)


class Trace:
    """Collector for trace spans with cost/latency roll-ups."""

    def __init__(self) -> None:
        self.spans: list[TraceSpan] = []

    # -- recording -----------------------------------------------------------
    def log(self, agent: str, message: str, **payload: Any) -> None:
        """Back-compatible plain event (zero cost/latency)."""
        self.spans.append(TraceSpan(agent=agent, op="log", message=message, payload=payload))

    def record(self, span: TraceSpan) -> None:
        self.spans.append(span)

    def record_llm(
        self, agent: str, op: str, response: LLMResponse, *, message: str = "llm call"
    ) -> TraceSpan:
        """Record an LLM call, computing cost from the response's usage/model."""
        span = TraceSpan(
            agent=agent,
            op=op,
            message=message,
            latency_ms=response.latency_ms,
            usage=response.usage,
            cost_usd=cost_for(response.model, response.usage),
            model=response.model,
            tool_calls=[t.name for t in response.tool_calls],
        )
        self.spans.append(span)
        return span

    # -- roll-ups ------------------------------------------------------------
    @property
    def events(self) -> list[TraceSpan]:  # back-compat alias
        return self.spans

    @property
    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for span in self.spans:
            total = total + span.usage
        return total

    @property
    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.spans), 6)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(s.latency_ms for s in self.spans), 3)

    # -- rendering -----------------------------------------------------------
    def as_markdown(self) -> str:
        usage = self.total_usage
        lines = [
            "# Agent Trace",
            "",
            f"- Total spans: {len(self.spans)}",
            f"- Total tokens: {usage.total_tokens} "
            f"(in {usage.input_tokens} / out {usage.output_tokens})",
            f"- Total cost: ${self.total_cost_usd:.6f}",
            f"- Total latency: {self.total_latency_ms:.1f} ms",
            "",
        ]
        for span in self.spans:
            lines.append(f"## {span.agent} · {span.op}")
            lines.append(f"- Time: `{span.timestamp}`")
            if span.message:
                lines.append(f"- Message: {span.message}")
            if span.model:
                lines.append(
                    f"- Model: `{span.model}` · {span.usage.total_tokens} tokens · "
                    f"${span.cost_usd:.6f} · {span.latency_ms:.1f} ms"
                )
            if span.tool_calls:
                lines.append(f"- Tool calls: {', '.join(span.tool_calls)}")
            if span.guardrail_events:
                lines.append(f"- Guardrails: {span.guardrail_events}")
            if span.payload:
                lines.append(f"- Payload: `{span.payload}`")
            lines.append("")
        return "\n".join(lines)


# Back-compat alias: the original class was named AgentTrace.
AgentTrace = Trace
