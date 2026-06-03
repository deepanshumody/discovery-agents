"""OpenAI GPT adapter (lazy import).

Install with ``pip install 'discovery-agents[openai]'`` and set OPENAI_API_KEY.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import RunConfig
from ._parsing import extract_json, json_instruction
from .types import LLMResponse, Message, TokenUsage, ToolInvocation, ToolSpec


class OpenAIClient:
    """LLMClient backed by OpenAI's Chat Completions API."""

    def __init__(self, model: str, config: RunConfig) -> None:
        from openai import OpenAI  # lazy import

        self.model = model
        self.config = config
        self._client = OpenAI(api_key=config.api_key) if config.api_key else OpenAI()

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: list[dict[str, Any]] = []
        for m in messages:
            role = m.role if m.role in ("system", "user", "assistant") else "user"
            content = m.content
            if role == "system" and response_format is not None:
                content += json_instruction(response_format)
            payload.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_object"}
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
        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0].message
        text = choice.content or ""
        tool_calls: list[ToolInvocation] = []
        for call in getattr(choice, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append(ToolInvocation(name=call.function.name, arguments=args, id=call.id))

        usage = TokenUsage(
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            structured=extract_json(text) if response_format is not None else None,
            usage=usage,
            latency_ms=latency_ms,
            model=self.model,
        )
