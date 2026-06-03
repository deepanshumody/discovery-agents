"""Google Gemini adapter (lazy import).

Install with ``pip install 'discovery-agents[gemini]'`` and set ``GEMINI_API_KEY``
(or pass a per-request key via ``RunConfig.api_key``). Uses the ``google-genai`` SDK.
Structured output is requested via ``response_mime_type="application/json"`` and parsed
with ``extract_json``. Tool-calling is not wired for Gemini (the pipeline agents use
structured output, not the ReAct tool loop).
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..config import RunConfig
from ._parsing import extract_json, json_instruction
from .types import LLMResponse, Message, TokenUsage, ToolSpec


class GeminiClient:
    """LLMClient backed by Google Gemini (google-genai)."""

    def __init__(self, model: str, config: RunConfig) -> None:
        from google import genai  # lazy import

        self.model = model
        self.config = config
        key = config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=key)

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        from google.genai import types  # lazy import

        system = "\n\n".join(m.content for m in messages if m.role == "system")
        if response_format is not None:
            system += json_instruction(response_format)
        prompt = "\n\n".join(m.content for m in messages if m.role != "system")

        config_kwargs: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if response_format is not None:
            config_kwargs["response_mime_type"] = "application/json"

        start = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = resp.text or ""
        usage_meta = getattr(resp, "usage_metadata", None)
        usage = TokenUsage(
            input_tokens=int(getattr(usage_meta, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage_meta, "candidates_token_count", 0) or 0),
        )
        return LLMResponse(
            text=text,
            tool_calls=[],
            structured=extract_json(text) if response_format is not None else None,
            usage=usage,
            latency_ms=latency_ms,
            model=self.model,
        )
