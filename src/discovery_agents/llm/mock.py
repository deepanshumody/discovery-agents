"""Deterministic, keyless mock LLM client.

This is the default provider: it lets the whole pipeline, the test suite, and CI
run with no API key and produce byte-for-byte reproducible output. It does not
fabricate domain answers (its `structured` is always None unless explicitly
scripted) — instead, each agent supplies a deterministic baseline that is used
when no real model produced valid output. The mock still returns realistic token
counts and a model id so observability spans are exercised.

For agent-loop tests, pass `script=[LLMResponse(...), ...]`; the mock returns
those responses in order (e.g. a scripted tool call followed by a final answer).
"""

from __future__ import annotations

import hashlib
from typing import Any

from .types import LLMResponse, Message, TokenUsage, ToolSpec


def _estimate_tokens(text: str) -> int:
    """Rough, deterministic token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


class MockLLMClient:
    """A deterministic stand-in for a real provider."""

    def __init__(self, script: list[LLMResponse] | None = None, model: str = "mock-1") -> None:
        self.model = model
        self._script: list[LLMResponse] | None = list(script) if script is not None else None
        self._cursor = 0

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self._script is not None and self._cursor < len(self._script):
            scripted = self._script[self._cursor]
            self._cursor += 1
            return scripted

        prompt = "\n".join(m.content for m in messages)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        text = f"[mock-llm:{digest}] deterministic response with no live model."
        return LLMResponse(
            text=text,
            tool_calls=[],
            structured=None,
            usage=TokenUsage(
                input_tokens=_estimate_tokens(prompt),
                output_tokens=_estimate_tokens(text),
            ),
            latency_ms=0.0,
            model=self.model,
        )
