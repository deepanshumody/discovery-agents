"""Provider-agnostic message and response types for the LLM layer.

These mirror the common surface of frontier chat APIs (Anthropic, Cohere, OpenAI)
so that agents depend on this small interface rather than any concrete SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single chat message. role is one of system|user|assistant|tool."""

    role: str
    content: str


@dataclass
class TokenUsage:
    """Token accounting for one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass
class ToolSpec:
    """A tool advertised to the model (name + JSON-schema parameters)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolInvocation:
    """A tool call requested by the model."""

    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass
class LLMResponse:
    """Normalized response from any provider.

    `structured` is a parsed JSON object when a `response_format` schema was
    requested and the model returned valid JSON; otherwise None.
    """

    text: str = ""
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    structured: dict[str, Any] | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    model: str = "mock-1"
