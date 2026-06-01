"""Output guardrails: citation-required, groundedness, and schema validation."""

from __future__ import annotations

from .base import GuardrailInput, GuardrailResult, Severity


class CitationRequiredGuard:
    """Fails when generated content carries no evidence citations."""

    name = "citation_required"

    def check(self, payload: GuardrailInput) -> GuardrailResult:
        if payload.citations:
            return GuardrailResult(self.name, passed=True, severity=Severity.OK)
        return GuardrailResult(
            self.name,
            passed=False,
            severity=Severity.BLOCK,
            reason="No evidence citations provided.",
        )


class GroundednessGuard:
    """Fails when any citation is not a real evidence id (hallucinated source)."""

    name = "groundedness"

    def check(self, payload: GuardrailInput) -> GuardrailResult:
        available = set(payload.available_evidence)
        if not available:
            # Cannot verify; treat as a non-blocking warning.
            return GuardrailResult(
                self.name,
                passed=True,
                severity=Severity.INFO,
                reason="No corpus to verify against.",
            )
        invalid = [cid for cid in payload.citations if cid not in available]
        if invalid:
            return GuardrailResult(
                self.name,
                passed=False,
                severity=Severity.BLOCK,
                reason=f"Citations not found in evidence: {', '.join(invalid)}",
            )
        return GuardrailResult(self.name, passed=True, severity=Severity.OK)


class SchemaGuard:
    """Fails when a structured payload is missing required keys."""

    name = "schema"

    def __init__(self, required_keys: list[str]) -> None:
        self.required_keys = required_keys

    def check(self, payload: GuardrailInput) -> GuardrailResult:
        missing = [key for key in self.required_keys if key not in payload.data]
        if missing:
            return GuardrailResult(
                self.name,
                passed=False,
                severity=Severity.WARN,
                reason=f"Missing required keys: {', '.join(missing)}",
            )
        return GuardrailResult(self.name, passed=True, severity=Severity.OK)
