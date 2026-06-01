"""Cohere Command adapter (lazy import).

Install with ``pip install 'discovery-agents[cohere]'`` and set COHERE_API_KEY.
Cohere's Command models are RAG- and tool-use-optimized, which fits this
pipeline's grounded, citation-first design.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import RunConfig
from ._parsing import extract_json, json_instruction
from .types import LLMResponse, Message, TokenUsage, ToolInvocation, ToolSpec

_ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"}


class CohereClient:
    """LLMClient backed by Cohere's Chat (v2) API."""

    def __init__(self, model: str, config: RunConfig) -> None:
        import cohere  # lazy import

        self.model = model
        self.config = config
        self._client = cohere.ClientV2()

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: list[dict[str, str]] = []
        for m in messages:
            content = m.content
            if m.role == "system" and response_format is not None:
                content += json_instruction(response_format)
            payload.append({"role": _ROLE_MAP.get(m.role, "user"), "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        start = time.perf_counter()
        resp = self._client.chat(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text_parts: list[str] = []
        message = getattr(resp, "message", None)
        for part in getattr(message, "content", None) or []:
            piece = getattr(part, "text", "")
            if piece:
                text_parts.append(str(piece))
        text = "".join(text_parts)

        tool_calls: list[ToolInvocation] = []
        for call in getattr(resp.message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments)
            except (json.JSONDecodeError, AttributeError, TypeError):
                args = {}
            tool_calls.append(
                ToolInvocation(name=call.function.name, arguments=args, id=getattr(call, "id", ""))
            )

        usage = TokenUsage(
            input_tokens=_usage_field(resp, "input_tokens"),
            output_tokens=_usage_field(resp, "output_tokens"),
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            structured=extract_json(text) if response_format is not None else None,
            usage=usage,
            latency_ms=latency_ms,
            model=self.model,
        )


def _usage_field(resp: Any, name: str) -> int:
    """Cohere nests token counts under usage.tokens.{input,output}_tokens."""
    usage = getattr(resp, "usage", None)
    tokens = getattr(usage, "tokens", None)
    value = getattr(tokens, name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
