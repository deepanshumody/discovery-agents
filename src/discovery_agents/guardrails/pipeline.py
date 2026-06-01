"""A guardrail pipeline that runs guards and records their decisions."""

from __future__ import annotations

from ..observability.trace import Trace, TraceSpan
from .base import Guardrail, GuardrailInput, GuardrailResult, Severity


class GuardrailPipeline:
    """Runs a set of guards over a payload and logs the outcome to the trace."""

    def __init__(
        self,
        guards: list[Guardrail],
        *,
        block_severity: Severity = Severity.BLOCK,
        trace: Trace | None = None,
    ) -> None:
        self.guards = guards
        self.block_severity = block_severity
        self.trace = trace

    def run(self, payload: GuardrailInput, *, stage: str = "guardrails") -> list[GuardrailResult]:
        results = [guard.check(payload) for guard in self.guards]
        if self.trace is not None:
            self.trace.record(
                TraceSpan(
                    agent="GuardrailPipeline",
                    op="guardrail",
                    message=stage,
                    guardrail_events=[r.as_event() for r in results],
                )
            )
        return results

    def is_blocked(self, results: list[GuardrailResult]) -> bool:
        return any((not r.passed) and r.severity >= self.block_severity for r in results)
