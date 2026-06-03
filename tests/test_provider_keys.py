"""Per-request API key handling and Gemini provider fallback."""

from __future__ import annotations

from discovery_agents.config import RunConfig
from discovery_agents.llm import MockLLMClient, get_client


def test_gemini_without_key_falls_back_to_mock(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert isinstance(get_client(RunConfig(provider="gemini")), MockLLMClient)


def test_request_api_key_avoids_mock_fallback(monkeypatch) -> None:
    # A per-request key means we do NOT fall back to mock even with no env var set.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = get_client(RunConfig(provider="anthropic", api_key="sk-test"))
    assert not isinstance(client, MockLLMClient)
    assert client.model == "claude-sonnet-4-6"


def test_gemini_with_key_constructs_gemini_client() -> None:
    from discovery_agents.llm.gemini_client import GeminiClient

    client = get_client(RunConfig(provider="gemini", api_key="test-key"))
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-2.0-flash"
