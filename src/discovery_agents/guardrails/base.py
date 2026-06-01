"""Guardrail protocol and shared types.

Guardrails make a run safe and auditable: input guards screen what enters the
pipeline (PII, prompt injection); output guards screen what leaves it (citations
present, citations grounded in real evidence, structured output well-formed).
Each guard returns a `GuardrailResult` with a severity, so a pipeline can flag or
block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable


class Severity(IntEnum):
    OK = 0
    INFO = 1
    WARN = 2
    BLOCK = 3


@dataclass
class GuardrailInput:
    """What a guard inspects."""

    text: str = ""
    citations: list[str] = field(default_factory=list)
    available_evidence: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailResult:
    name: str
    passed: bool
    severity: Severity = Severity.OK
    reason: str = ""
    redacted_text: str | None = None

    def as_event(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity.name,
            "reason": self.reason,
        }


@runtime_checkable
class Guardrail(Protocol):
    name: str

    def check(self, payload: GuardrailInput) -> GuardrailResult: ...
