"""Anthropic Claude adapter (the default real provider).

Imported lazily so the core package has no hard dependency on the SDK. Install
with ``pip install 'discovery-agents[anthropic]'`` and set ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import RunConfig
from ._parsing import extract_json, json_instruction
from .types import LLMResponse, Message, TokenUsage, ToolInvocation, ToolSpec


class AnthropicClient:
    """LLMClient backed by Anthropic's Messages API."""

    def __init__(self, model: str, config: RunConfig) -> None:
        from anthropic import Anthropic  # lazy import

        self.model = model
        self.config = config
        self._client = Anthropic()

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        if response_format is not None:
            system_text += json_instruction(response_format)

        conversation = [
            {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": conversation,
        }
        if system_text:
            kwargs["system"] = system_text
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        start = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text_parts: list[str] = []
        tool_calls: list[ToolInvocation] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolInvocation(name=block.name, arguments=dict(block.input), id=block.id)
                )
        text = "".join(text_parts)

        usage = TokenUsage(
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            structured=extract_json(text) if response_format is not None else None,
            usage=usage,
            latency_ms=latency_ms,
            model=self.model,
        )
