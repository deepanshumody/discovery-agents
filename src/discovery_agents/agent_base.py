"""Agent base class and tracing re-exports.

The trace types now live in `observability.trace`; they are re-exported here for
backward compatibility. `BaseAgent` gains an `LLMClient` (default: the keyless
mock) and a `_chat` helper that records a fully-costed trace span per call.
"""

from __future__ import annotations

import logging
from typing import Any

from .llm import LLMClient, LLMResponse, Message, MockLLMClient, ToolSpec
from .observability.trace import AgentTrace, Trace, TraceSpan

logger = logging.getLogger("discovery_agents.agents")

__all__ = ["AgentTrace", "BaseAgent", "Trace", "TraceSpan"]


class BaseAgent:
    """Small base class so every agent has a name, a trace, and an LLM client."""

    name = "BaseAgent"

    def __init__(self, trace: Trace | None = None, llm: LLMClient | None = None) -> None:
        self.trace: Trace = trace or Trace()
        self.llm: LLMClient = llm or MockLLMClient()

    def log(self, message: str, **payload: Any) -> None:
        self.trace.log(self.name, message, **payload)

    def _chat(
        self,
        *,
        op: str,
        system: str,
        user: str,
        schema: dict[str, Any] | None = None,
        tools: list[ToolSpec] | None = None,
        message: str = "llm call",
    ) -> LLMResponse:
        """Run one LLM turn and record a costed trace span.

        With the mock client this returns a deterministic response whose
        `structured` is None, so callers fall back to their deterministic
        baseline. With a real provider, the same prompt is sent to the model.
        """
        messages = [Message(role="system", content=system), Message(role="user", content=user)]
        try:
            response = self.llm.chat(messages, tools=tools, response_format=schema)
        except Exception as exc:
            # A provider error (429, network, bad key, ...) must never crash a run — degrade
            # to the agent's deterministic fallback (structured=None) and record the failure.
            logger.warning(
                "LLM call failed in %s (%s); falling back to deterministic.",
                self.name,
                type(exc).__name__,
            )
            self.trace.record(
                TraceSpan(agent=self.name, op=op, message=f"llm-error: {type(exc).__name__}")
            )
            return LLMResponse(structured=None, model=getattr(self.llm, "model", "unknown"))
        self.trace.record_llm(self.name, op, response, message=message)
        return response
