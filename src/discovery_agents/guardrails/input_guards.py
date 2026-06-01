"""Input guardrails: PII redaction and prompt-injection detection."""

from __future__ import annotations

import re

from .base import GuardrailInput, GuardrailResult, Severity

# Order matters: redact 16-digit cards before SSN/phone so sub-spans aren't mislabeled.
# Regex PII detection is best-effort, not a compliance control.
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CARD", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
]

# Heuristic injection detection (defense in depth, not a guarantee): verbs may sit a few
# words away from their target, singular/plural forms, and paraphrases are all covered.
_DOT = re.IGNORECASE | re.DOTALL
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}?\b"
        r"(instruction|instructions|prompt|prompts|directive|directives|rule|rules)\b",
        _DOT,
    ),
    re.compile(
        r"\b(reveal|show|print|leak|repeat)\b.{0,30}?\b(system\s+)?(prompt|instructions)\b", _DOT
    ),
    re.compile(r"\boverride\b.{0,20}?\b(rules|guardrails|safety|restrictions)\b", _DOT),
    re.compile(r"\byou are now\b", re.IGNORECASE),
    re.compile(r"\bpay no attention to\b", re.IGNORECASE),
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
        # Normalize so whitespace/newlines and zero-width chars can't split tokens.
        cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", payload.text)
        normalized = re.sub(r"\s+", " ", cleaned)
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(normalized):
                return GuardrailResult(
                    self.name,
                    passed=False,
                    severity=Severity.BLOCK,
                    reason=f"Possible prompt injection: matched {pattern.pattern!r}",
                )
        return GuardrailResult(self.name, passed=True, severity=Severity.OK)
