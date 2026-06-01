"""Construct an LLMClient from a RunConfig, with graceful keyless fallback."""

from __future__ import annotations

import logging
import os

from ..config import RunConfig
from .base import LLMClient
from .mock import MockLLMClient

logger = logging.getLogger("discovery_agents.llm")


def get_client(config: RunConfig | None = None) -> LLMClient:
    """Return a provider client, or the deterministic mock if unavailable.

    Falls back to MockLLMClient (with a logged warning) when the requested
    provider is unknown, its API key is unset, or its SDK is not installed.
    This guarantees the pipeline always runs, key or no key.
    """
    config = config or RunConfig.from_env()
    provider = config.provider
    model = config.resolved_model()

    if provider in ("", "mock"):
        return MockLLMClient(model="mock-1")

    key_env = config.key_env_var()
    if key_env and not os.environ.get(key_env):
        logger.warning(
            "Provider %r selected but %s is not set; falling back to the mock client.",
            provider,
            key_env,
        )
        return MockLLMClient(model="mock-1")

    try:
        if provider == "anthropic":
            from .anthropic_client import AnthropicClient

            return AnthropicClient(model=model, config=config)
        if provider == "cohere":
            from .cohere_client import CohereClient

            return CohereClient(model=model, config=config)
        if provider == "openai":
            from .openai_client import OpenAIClient

            return OpenAIClient(model=model, config=config)
    except ImportError as exc:  # SDK not installed
        logger.warning(
            "Provider %r SDK not installed (%s); falling back to the mock client. "
            "Install with: pip install 'discovery-agents[%s]'",
            provider,
            exc,
            provider,
        )
        return MockLLMClient(model="mock-1")

    logger.warning("Unknown provider %r; falling back to the mock client.", provider)
    return MockLLMClient(model="mock-1")
