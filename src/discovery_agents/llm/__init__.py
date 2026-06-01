"""Provider-agnostic LLM layer.

Agents depend only on the `LLMClient` protocol and the message/response types.
The deterministic `MockLLMClient` is the keyless default; real adapters are
imported lazily via `get_client`.
"""

from __future__ import annotations

from .base import LLMClient
from .factory import get_client
from .mock import MockLLMClient
from .types import LLMResponse, Message, TokenUsage, ToolInvocation, ToolSpec

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "MockLLMClient",
    "TokenUsage",
    "ToolInvocation",
    "ToolSpec",
    "get_client",
]
