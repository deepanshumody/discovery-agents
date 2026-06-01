"""Input guardrails: PII redaction and prompt-injection detection."""

from __future__ import annotations

import re

from .base import GuardrailInput, GuardrailResult, Severity

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
]

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore (the )?(previous|above|prior) (instructions|prompt)", re.IGNORECASE),
    re.compile(r"disregard (the )?(previous|above|system)", re.IGNORECASE),
    re.compile(r"reveal (your|the) (system )?(prompt|instructions)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"override (the )?(rules|guardrails|safety)", re.IGNORECASE),
]


class PiiRedactionGuard:
    """Redacts emails, phone numbers, and SSNs; warns but does not block."""

    name = "pii_redaction"

    def check(self, payload: GuardrailInput) -> GuardrailResult:
        text = payload.text
        found: list[str] = []
        for label, pattern in _PII_PATTERNS:
            if pattern.search(text):
                found.append(label)
                text = pattern.sub(f"[REDACTED_{label}]", text)
        if not found:
            return GuardrailResult(self.name, passed=True, severity=Severity.OK)
        return GuardrailResult(
            self.name,
            passed=True,  # redaction resolves it; the run may continue
            severity=Severity.WARN,
            reason=f"Redacted PII: {', '.join(sorted(set(found)))}",
            redacted_text=text,
        )


class PromptInjectionGuard:
    """Flags prompt-injection attempts as a blocking violation."""

    name = "prompt_injection"

    def check(self, payload: GuardrailInput) -> GuardrailResult:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(payload.text):
                return GuardrailResult(
                    self.name,
                    passed=False,
                    severity=Severity.BLOCK,
                    reason=f"Possible prompt injection: matched {pattern.pattern!r}",
                )
        return GuardrailResult(self.name, passed=True, severity=Severity.OK)
