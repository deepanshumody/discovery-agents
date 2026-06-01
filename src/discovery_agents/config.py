"""Run configuration for the discovery pipeline.

Resolved from explicit arguments or environment variables. Keyless-by-default:
if no provider is set (or the selected provider's key/SDK is missing), the
pipeline falls back to the deterministic mock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default model per provider. Override with DISCOVERY_MODEL or --model.
DEFAULT_MODELS: dict[str, str] = {
    "mock": "mock-1",
    "anthropic": "claude-sonnet-4-6",
    "cohere": "command-r-plus",
    "openai": "gpt-4o",
}

# Environment variable that holds each provider's API key.
PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "cohere": "COHERE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@dataclass
class RunConfig:
    """Everything needed to execute one pipeline run."""

    provider: str = "mock"
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024
    max_steps: int = 6
    token_budget: int | None = None
    use_rag: bool = True
    use_langgraph: bool = False

    @classmethod
    def from_env(cls) -> RunConfig:
        """Build a config from DISCOVERY_* environment variables."""
        return cls(
            provider=os.environ.get("DISCOVERY_PROVIDER", "mock").lower(),
            model=os.environ.get("DISCOVERY_MODEL") or None,
            temperature=float(os.environ.get("DISCOVERY_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("DISCOVERY_MAX_TOKENS", "1024")),
            max_steps=int(os.environ.get("DISCOVERY_MAX_STEPS", "6")),
            use_rag=os.environ.get("DISCOVERY_USE_RAG", "1") != "0",
            use_langgraph=os.environ.get("DISCOVERY_USE_LANGGRAPH", "0") == "1",
        )

    def resolved_model(self) -> str:
        """The concrete model id to use for this run."""
        if self.model:
            return self.model
        return DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS["mock"])

    def key_env_var(self) -> str | None:
        """Name of the env var that must hold the provider's key, if any."""
        return PROVIDER_KEY_ENV.get(self.provider)
