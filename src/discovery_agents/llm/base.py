"""The LLMClient protocol every provider adapter implements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import LLMResponse, Message, ToolSpec


@runtime_checkable
class LLMClient(Protocol):
    """Minimal chat interface shared by every provider (and the mock).

    Implementations must be side-effect-free apart from the network call, must
    populate `usage`/`latency_ms`/`model` on the response, and must never raise
    for a missing optional dependency at import time (import the SDK lazily).
    """

    model: str

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a chat request and return a normalized response.

        - `tools`: tool specs the model may call (returned as `tool_calls`).
        - `response_format`: a JSON schema. When provided, the adapter asks the
          model for JSON and parses it into `LLMResponse.structured`.
        """
        ...
