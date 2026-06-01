"""Helpers for coaxing structured JSON out of free-form model text."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a JSON object from model output.

    Tries the whole string, then a fenced code block, then the first balanced
    ``{...}`` span. Returns None if nothing parses to a dict.
    """
    if not text:
        return None

    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"items": value}
    return None


def _candidates(text: str) -> list[str]:
    out = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        out.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        out.append(text[start : end + 1])
    return out


def json_instruction(schema: dict[str, Any]) -> str:
    """A system-prompt suffix instructing the model to emit schema-shaped JSON."""
    return (
        "\n\nRespond with a single valid JSON object and nothing else. "
        "It must conform to this JSON schema:\n"
        f"{json.dumps(schema)}"
    )
